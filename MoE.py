import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    Each expert is a small feed-forward subnetwork.
    Here, we map d_model -> d_ff -> d_model.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x: [B, T, d_model]
        return self.net(x)


class MMoEHead(nn.Module):
    """
    Shared Expert Pool + 2 Gating Networks (one for each task).
    Then each task does a final linear layer to get the actual prediction.
    """
    def __init__(self, d_model, d_ff, n_expert, d_long):
        """
        d_model: dimension of transformer output
        d_ff: hidden dimension inside each expert
        n_expert: number of experts
        d_long: dimension of the longitudinal output
        """
        super().__init__()

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(n_expert)
        ])

        self.gate_long = nn.Linear(d_model, n_expert)
        self.gate_surv = nn.Linear(d_model, n_expert)

        self.long_out = nn.Linear(d_model, d_long)  
        self.surv_out = nn.Linear(d_model, 1)       

    def forward(self, x):
        """
        x: [B, T, d_model] from the Transformer.
        Returns:
          long_pred: [B, T, d_long]
          surv_pred: [B, T, 1]
        """
        B, T, _ = x.shape

        expert_outs = []
        for expert in self.experts:
            e_out = expert(x)             # [B, T, d_model]
            expert_outs.append(e_out)

        expert_outs = torch.stack(expert_outs, dim=2)  #[B, T, d_model] => [B, T, n_expert, d_model]

        
        gate_logits_long = self.gate_long(x)              # [B, T, n_expert]
        gate_weights_long = F.softmax(gate_logits_long, dim=-1)  
        
        gate_weights_long = gate_weights_long.unsqueeze(-1)       # [B, T, n_expert, 1]


        long_combined = (expert_outs * gate_weights_long).sum(dim=2) #=> [B, T, d_model]


        gate_logits_surv = self.gate_surv(x)              # [B, T, n_expert]
        gate_weights_surv = F.softmax(gate_logits_surv, dim=-1)
        gate_weights_surv = gate_weights_surv.unsqueeze(-1)       # [B, T, n_expert, 1]

        surv_combined = (expert_outs * gate_weights_surv).sum(dim=2)


        long_pred = self.long_out(long_combined)    # [B, T, d_long]
        surv_logit = self.surv_out(surv_combined)    # [B, T, 1]
        surv_pred = torch.sigmoid(surv_logit)        

        return long_pred, surv_pred