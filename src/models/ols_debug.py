import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tensordict import TensorDict
import gc

from infrastructure import utils


class CnnKF(nn.Module):
    def __init__(self, ny: int, ir_length: int, ridge: float = 0.0):
        super().__init__()

        self.O_D = ny
        self.ir_length = ir_length

        self.observation_IR = nn.Parameter(torch.zeros(self.O_D, self.ir_length, self.O_D))  # [O_D x R x O_D]

        r_ = self.ir_length * self.O_D
        self.XTX = ridge * torch.eye(r_)  # torch.zeros((r_, r_))                                # [RO_D x RO_D]
        self.XTy = torch.zeros((r_, self.O_D))  # [RO_D x O_D]

        self.X = torch.zeros((0, r_))  # [k x RO_D]
        self.y = torch.zeros((0, self.O_D))  # [k x O_D]


    
    @classmethod
    def analytical_error(cls, observation_IR: torch.Tensor, systems: TensorDict[str, torch.Tensor], ridge = 0.0) -> torch.Tensor:
        with torch.no_grad():
            device = observation_IR.device

            # Variable definition
            Q = utils.complex(observation_IR)  # [B... x O_D x R x O_D]
            Q = Q.permute(*range(Q.ndim - 3), -2, -1, -3)  # [B... x R x O_D x O_D]

            F = utils.complex(systems["F"])  # [B... x S_D x S_D]
            H = utils.complex(systems["H"])  # [B... x O_D x S_D]
            sqrt_S_W = utils.complex(systems["sqrt_S_W"])  # [B... x S_D x S_D]
            sqrt_S_V = utils.complex(systems["sqrt_S_V"])  # [B... x O_D x O_D]

            R = Q.shape[-3]

            L, V = torch.linalg.eig(F)  # [B... x S_D], [B... x S_D x S_D]
            Vinv = torch.linalg.inv(V)  # [B... x S_D x S_D]

            Hs = H @ V  # [B... x O_D x S_D]
            sqrt_S_Ws = Vinv @ sqrt_S_W  # [B... x S_D x S_D]

            # State evolution noise error
            # Highlight
            ws_current_err = (Hs @ sqrt_S_Ws).norm(dim=(-2, -1)) ** 2  # [B...]
            
            L_pow_series = L.unsqueeze(-2) ** torch.arange(1, R + 1)[:, None]  # [B... x R x S_D]
            L_pow_series_inv = 1. / L_pow_series  # [B... x R x S_D]

            QlHsLl = (Q.to(device) @ Hs.unsqueeze(-3).to(device)) * L_pow_series_inv.unsqueeze(-2).to(device)  # [B... x R x O_D x S_D]
            Hs_cumQlHsLl = Hs.unsqueeze(-3).to(device) - torch.cumsum(QlHsLl, dim=-3).to(device)  # [B... x R x O_D x S_D]
            Hs_cumQlHsLl_Lk = Hs_cumQlHsLl.to(device) * L_pow_series.unsqueeze(-2).to(device)  # [B... x R x O_D x S_D]

            # Highlight
            ws_recent_err = (Hs_cumQlHsLl_Lk.to(device) @ sqrt_S_Ws.to(device).unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2  # [B...]

            Hs_cumQlHsLl_R = Hs_cumQlHsLl.index_select(-3, torch.tensor([R - 1]).to(device)).squeeze(-3).to(device)  # [B... x O_D x S_D]
            cll = L.unsqueeze(-1).to(device) * L.unsqueeze(-2).to(device)  # [B... x S_D x S_D]

            # Highlight
            _ws_geometric = (Hs_cumQlHsLl_R.mT @ Hs_cumQlHsLl_R) * ((cll ** (R + 1)) / (1 - cll).to(device))  # [B... x S_D x S_D]
            ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT.to(device) @ _ws_geometric.to(device) @ sqrt_S_Ws.to(device))  # [B...]

            # Observation noise error
            # Highlight
            v_current_err = sqrt_S_V.norm(dim=(-2, -1)) ** 2  # [B...]

            # #check if Q is zero
            # if torch.all(Q == 0):
            #     print("Q is zero")
            # Highlight
            # TODO: Backward pass on the above one breaks when Q = 0 for some reason
            # v_recent_err = (Q @ sqrt_S_V.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2           # [B...]
            v_recent_err = utils.batch_trace(sqrt_S_V.mT.to(device) @ (Q.mT.to(device) @ Q.to(device)).sum(dim=-3) @ sqrt_S_V.to(device))  # [B...]

            err = ws_current_err.to(device) + ws_recent_err.to(device) + ws_geometric_err.to(device) + v_current_err.to(device) + v_recent_err.to(device)  # [B...]
            del ws_current_err
            del ws_geometric_err
            del ws_recent_err
            del QlHsLl
            del Hs_cumQlHsLl
            del Hs_cumQlHsLl_Lk
            del Hs_cumQlHsLl_R
            del v_current_err
            del v_recent_err
            del Q
            del L 
            del V 
            del L_pow_series
            del L_pow_series_inv
            del observation_IR
            torch.cuda.empty_cache()
            gc.collect()

        return err.real

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns {
            'observation_estimation': [B x L x O_D]
        }
    """

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # [5, 2, 5] [1, 5, 2, 1]
        return Fn.conv2d(
            self.observation_IR,
            context.transpose(-2, -1).unsqueeze(-1).flip(-2),
        )

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns None
    """

    def update(self,
               context: torch.Tensor,  # [B... x R x O_D]
               target: torch.Tensor  # [B... x O_D]
               ):
        # DONE: Implement online least squares for memory efficiency
        flattened_X = context.flip(-2).view((-1, self.ir_length * self.O_D))  # [B x RO_D]
        flattened_observations = target.view((-1, self.O_D))  # [B x O_D]

        self.XTX = self.XTX + (flattened_X.mT @ flattened_X)
        self.XTy = self.XTy + (flattened_X.mT @ flattened_observations)

        if torch.linalg.matrix_rank(self.XTX) >= self.XTX.shape[0]:
            flattened_w = torch.linalg.inv(self.XTX) @ self.XTy  # [RO_D x RO_D]
        else:
            self.X = torch.cat([self.X, flattened_X], dim=0)
            self.y = torch.cat([self.y, flattened_observations], dim=0)
            flattened_w = torch.linalg.pinv(self.X) @ self.y

        self.observation_IR.data = flattened_w.unflatten(0, (self.ir_length, -1)).transpose(0, 1)  # [O_D x R x O_D]
        return self.observation_IR.data