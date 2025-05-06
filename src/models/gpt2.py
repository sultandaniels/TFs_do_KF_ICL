import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, TransfoXLConfig, TransfoXLModel
from models.lightning_base_model import BaseModel
from core import Config
from linalg_helpers import print_matrix
import pickle

config = Config()


class GPT2(BaseModel):
    def __init__(self, n_dims_in, n_positions, n_embd, n_layer=12, n_head=8, n_dims_out=5, learning_rate=config.learning_rate, use_pos_emb=config.use_pos_emb):
        super(GPT2, self).__init__(learning_rate=learning_rate)
        gpt_configuration = GPT2Config(
            use_pos_emb=use_pos_emb,
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

            if config.mem_suppress and config.masking:
                #create a mask to identify rows of ys that are all zeros and also all indices from the list of lists input_dict["mask_idx"]
                #ys is of shape [batch_size, seq_len, dims]
                mask_all_zeros = torch.all(ys == 0, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
                
                mask_selected_indices = torch.zeros_like(mask_all_zeros, dtype=torch.bool)
                for b, idx_list in enumerate(input_dict["mask_idx"]):
                    mask_idx_minus_1 = [int(idx) - 1 for idx in idx_list]
                    mask_selected_indices[b, mask_idx_minus_1, :] = True #since the target has one less entry than the full seqment we need to subtract 1 from the index

                mask = mask_all_zeros | mask_selected_indices #combine the two masks with a logical OR

            else:
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
