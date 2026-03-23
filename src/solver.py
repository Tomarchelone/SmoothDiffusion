import torch
import torch.nn as nn

class Solver(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, m):
        # x (1, H)
        # m (L, H, H), L = power of two

        # -> (L + 1, H) - The result of cumulative multiplication of x on m. Initial x is also returned

        m_buffer = [m]

        cur_m = m
        while cur_m.size(0) != 1:
            # cur_m (L, H, H)

            left = cur_m[::2]
            right = cur_m[1::2]

            cur_m = torch.bmm(left, right)

            if cur_m.size(0) != 1:
                m_buffer.append(cur_m)
        
        cur = x.unsqueeze(0) # 1, 1, H

        last = torch.bmm(cur, cur_m) # 1, 1, H
        
        for idx in range(len(m_buffer)):
            # cur (L, 1, H)
            m = m_buffer[-idx - 1] # L * 2, H, H

            m = m[::2] # L, H, H

            cur2 = torch.bmm(cur, m) # L, 1, H

            cur = torch.cat((
                cur.unsqueeze(1)
                , cur2.unsqueeze(1)
                ), dim=1
            ).flatten(0, 1) # L * 2, 1, H


        res = torch.cat(
            (cur[:, 0, :] # L, H
            , last[:, 0, :]) # 1, H
            , dim=0
        )

        return res # L + 1, H