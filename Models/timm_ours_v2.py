import torch
import torch.nn as nn
import timm

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
        dropout=0.1,
        second_path_size=None,
        print_w=False,
        drop_path=0.1,
        attention_method='default'
    ):
        super(VisionTransformer, self).__init__()

        vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        self.vit_blocks = vit.blocks            
        self.vit_norm = vit.norm                
        self.vit_pos_drop = vit.pos_drop        
        self.vit_head = vit.head                

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)  
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3,   kernel_size=4,  stride=4)   
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=4,  stride=4)   

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, 768))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, 768))

        self.pos_embed1 = nn.Parameter(torch.zeros(1, 197, 768))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 197, 768))

        # Init tokens / pos embeds ##################################################################
        # Init tokens / pos embeds ##################################################################
        nn.init.trunc_normal_(self.cls_token1, std=0.02)
        nn.init.trunc_normal_(self.cls_token3, std=0.02)
        nn.init.trunc_normal_(self.pos_embed1, std=0.02)
        nn.init.trunc_normal_(self.pos_embed3, std=0.02)
        # Init tokens / pos embeds ##################################################################
        # Init tokens / pos embeds ##################################################################

        self.mlp_intermediate = nn.Sequential(
            nn.LayerNorm(768 * 2),
            nn.Linear(768 * 2, 768),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        for p in self.parameters():
            p.requires_grad = False

        for p in self.mlp_intermediate.parameters():
            p.requires_grad = True

        for p in self.vit_head.parameters():
            p.requires_grad = True

    def forward(self, x):
        feat1 = self.conv1(x)                   
        feat3 = self.conv3(self.conv2(x))        

        B = x.size(0)

        feat1 = feat1.flatten(2).transpose(1, 2)  
        feat3 = feat3.flatten(2).transpose(1, 2)  

        cls1 = self.cls_token1.expand(B, -1, -1)  
        cls3 = self.cls_token3.expand(B, -1, -1)  

        feat1 = torch.cat([cls1, feat1], dim=1)  
        feat3 = torch.cat([cls3, feat3], dim=1)  

        # feat1 = self.vit_pos_drop(feat1 + self.pos_embed1)
        # feat3 = self.vit_pos_drop(feat3 + self.pos_embed3)
        feat1 = feat1 + self.pos_embed1
        feat3 = feat3 + self.pos_embed3

        out1 = self.vit_blocks(feat1)            
        out3 = self.vit_blocks(feat3)            

        out1 = self.vit_norm(out1)
        out3 = self.vit_norm(out3)

        cls1 = out1[:, 0]                       
        cls3 = out3[:, 0]                         
        combined = torch.cat([cls1, cls3], dim=1) 

        fused = self.mlp_intermediate(combined)  

        logits = self.vit_head(fused)
        return logits

