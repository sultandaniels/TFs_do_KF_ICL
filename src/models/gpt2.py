import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, TransfoXLConfig, TransfoXLModel
from models import BaseModel
from core import Config
from linalg_helpers import print_matrix

config = Config()

class GPT2Parent(GPT2Model):
    def __init__(self, config):
        super(GPT2Parent, self).__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class GPT2(BaseModel):
    def __init__(self, n_dims_in, n_positions, n_embd, n_layer=12, n_head=8, n_dims_out=5, learning_rate=config.learning_rate):
        super(GPT2, self).__init__(learning_rate=learning_rate)
        gpt_configuration = GPT2Config(
            n_positions=2048,  # set to sthg large advised
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        
        # olmo_configuration = OlmoConfig(
        #     vocab_size=50304,
        #     hidden_size=4096,
        #     intermediate_size=11008,
        #     num_hidden_layers=32,
        #     num_attention_heads=32,
        #     max_position_embeddings=2048,
        #     use_cache=False
        # )

        self.n_positions = n_positions
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self._read_in = nn.Linear(n_dims_in, n_embd)

        if config.model_type == "GPT2":
            self._backbone = GPT2Model(gpt_configuration)
            self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        elif config.model_type == "GPT2_NoPE":
            print()
            self._backbone = GPT2Model(gpt_configuration)
            self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        elif config.model_type == "transfoXL":
            self._backbone = TransfoXLModel(gpt_configuration)
            self.name = f"transfoxl_embd={n_embd}_layer={n_layer}_head={n_head}"
        # elif config.model_type == "olmo":
        #     self._backbone = OlmoModel(olmo_configuration)

        self._read_out = nn.Linear(n_embd, n_dims_out)

    def predict_step(self, input_dict, batch_idx=None):
        current = input_dict["current"]
        embeds = self._read_in(current)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        # predict only on xs
        return input_dict, {"preds": prediction}

    def forward(self, input_dict, batch_idx=None, return_intermediate_dict=False):
        input_dict, intermediate_dict = self.predict_step(input_dict, batch_idx)

        # Calculate all loss terms and metrics (scores)
        output_dict = self.calculate_losses_and_metrics(input_dict, intermediate_dict)

        # Calculate optimized loss
        optimized_loss = 0
        for key, loss in output_dict.items():
            if "loss_" in key:
                optimized_loss += loss
        output_dict["optimized_loss"] = optimized_loss
        return (intermediate_dict, output_dict) if return_intermediate_dict else output_dict

    def calculate_losses_and_metrics(self, input_dict, intermediate_dict):
        
        # Calculate loss
        ys = input_dict["target"]

        preds = intermediate_dict["preds"]
        res_sq = (preds - ys) ** 2 #residuals squared

        if config.multi_sys_trace:
            # Create a mask to identify rows of ys that are all zeros
            mask = torch.all(ys == 0, dim=-1, keepdim=True)
            
            # Apply the mask to res_sq to disregard the residuals for rows of ys that are all zeros
            res_sq = res_sq.masked_fill(mask, 0)

            output_dict = {"loss_mse": torch.sum(res_sq) / (~mask).sum()} #mean squared error loss
        else:
            output_dict = {"loss_mse": torch.mean(res_sq)}
            

        # Calculate metrics
        for i in range(ys.shape[1]):
            for j in range(ys.shape[2]):
                output_dict[f"metric_mse_ts{i}_dim_{j}"] = torch.mean(res_sq[:, i, j])

        return output_dict

    def predict_ar(self, ins, fix_window_len=True):
        ins = torch.from_numpy(ins).float().to(self.device)
        one_d = False
        if ins.ndim == 2:
            one_d = True
            ins = ins.unsqueeze(0)
        bsize, points, _ = ins.shape
        d_o = self.n_dims_out
        outs = torch.zeros(bsize, 1, d_o).to(self.device)
        with torch.no_grad():
            for i in range(1, points + 1):
                I = ins[:, :i]
                if fix_window_len and I.shape[1] > self.n_positions:
                    I = I[:, -self.n_positions:]
                _, interm = self.predict_step({"xs": I})
                pred = interm["preds"][:, -1:]  # b, 1, d_o
                outs = torch.cat([outs, pred], dim=1)
        outs = outs.detach().cpu().numpy()
        if one_d:
            outs = outs[0]
        return outs
