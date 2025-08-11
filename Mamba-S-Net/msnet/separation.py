import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

class FeatureConversion(nn.Module):
    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse = inverse
        self.channels = channels

    def forward(self, x):
        if self.inverse:
            x = x.float()
            x_r = x[:, :self.channels//2, :, :]
            x_i = x[:, self.channels//2:, :, :]
            x = torch.complex(x_r, x_i)
            x = torch.fft.irfft(x, dim=3, norm="ortho")
        else:
            x = x.float()
            x = torch.fft.rfft(x, dim=3, norm="ortho")
            x_real = x.real
            x_imag = x.imag
            x = torch.cat([x_real, x_imag], dim=1)
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.dt_min = dt_min
        self.dt_max = dt_max

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        B, L, D = x.shape

        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=self.d_inner, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = self.act(x)

        y = self.ssm(x)

        y = y * self.act(res)
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        B, L, d_inner = x.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        dt, B_proj, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt + self.dt_proj.bias)

        y = self.selective_scan(x, dt, A, B_proj, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        B_batch, L, d_inner = u.shape
        d_state = A.shape[1]

        x = torch.zeros((B_batch, d_inner, d_state), device=u.device, dtype=u.dtype)
        ys = []

        for i in range(L):
            delta_i = delta[:, i]
            u_i = u[:, i]
            B_i = B[:, i]
            C_i = C[:, i]

            deltaA_i = torch.exp(delta_i.unsqueeze(-1) * A.unsqueeze(0))

            deltaB_u_i = delta_i.unsqueeze(-1) * B_i.unsqueeze(1) * u_i.unsqueeze(-1)

            x = deltaA_i * x + deltaB_u_i

            y_i = torch.sum(x * C_i.unsqueeze(1), dim=-1)
            ys.append(y_i)

        y = torch.stack(ys, dim=1)

        y = y + u * D.unsqueeze(0).unsqueeze(0)

        return y


class DualPathMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1):
        super().__init__()

        self.d_model = d_model

        self.freq_mamba = MambaBlock(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max)
        self.time_mamba = MambaBlock(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max)

        self.freq_norm = nn.LayerNorm(d_model)
        self.time_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, F, T = x.shape

        x_freq = x.permute(0, 3, 2, 1).contiguous().view(B * T, F, C)

        x_freq_out = self.freq_mamba(x_freq)

        x_freq_out = self.freq_norm(x_freq + x_freq_out)

        x = x_freq_out.view(B, T, F, C).permute(0, 3, 2, 1)

        x_time = x.permute(0, 2, 3, 1).contiguous().view(B * F, T, C)

        x_time_out = self.time_mamba(x_time)

        x_time_out = self.time_norm(x_time + x_time_out)

        x = x_time_out.view(B, F, T, C).permute(0, 3, 1, 2)

        return x


class SeparationNet(nn.Module):
    def __init__(self, channels, expand=2, num_layers=6, d_state=16, d_conv=4,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1):
        super(SeparationNet, self).__init__()

        self.num_layers = num_layers

        self.dp_modules = nn.ModuleList([
            DualPathMamba(
                d_model=channels * (2 if i % 2 == 1 else 1),
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max
            ) for i in range(num_layers)
        ])

        self.feature_conversion = nn.ModuleList([
            FeatureConversion(channels * 2, inverse = False if i % 2 == 0 else True) for i in range(num_layers)
        ])

    def forward(self, x):
        for i in range(self.num_layers):
            if self.training:
                x = checkpoint.checkpoint(self.dp_modules[i], x, use_reentrant=False)
                x = checkpoint.checkpoint(self.feature_conversion[i], x, use_reentrant=False)
            else:
                x = self.dp_modules[i](x)
                x = self.feature_conversion[i](x)
        return x
