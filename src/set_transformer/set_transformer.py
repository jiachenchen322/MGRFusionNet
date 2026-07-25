import torch 
from torch import nn 
from torch.nn import functional as F
from copy import deepcopy

class rFF(nn.Module):
    @property 
    def dim_ff(self,):
        return self.lin2.out_features  
    
    def __init__(self, in_dims, dim_ff=100):
        super().__init__()
        self.lin1 = nn.Linear(in_dims, dim_ff) 
        self.lin2 = nn.Linear(dim_ff, in_dims) 

    def forward(self, x):
        """
        x -> (batchsize, n, d) 
        """
        y = self.lin1(x) 
        y = F.relu(y) 
        y = self.lin2(y)
        return y

class EncoderLayer(nn.Module):
    def __init__(self, num_features, num_heads=1, dim_ff=100, norm_first=False):
        """
        Arguments - 
        num_features <int> - The dimension of the embedding from the graph GNN
        num_heads <int> - Number of 
        """
        super().__init__()
        #self.mha = nn.MultiheadAttention(embed_dim=num_features, num_heads=num_heads, bias=False, batch_first=True)
        self.mha = nn.MultiheadAttention(embed_dim=num_features, num_heads=num_heads, bias=False) # our version does not support batch_first param
        self.ln1 = nn.LayerNorm(num_features)
        self.rff = rFF(num_features, dim_ff)
        self.ln2 = nn.LayerNorm(num_features)
        self.norm_first = norm_first
    
    def norm_first_forward(self, x, need_weights):
        #  self attention sublayer  
        x = self.ln1(x)
        out = self.mha(x, x, x, need_weights=need_weights)
        if need_weights:
            out, attn_weights = out
        else:
            if type(out) == tuple:
                out = out[0]
        x = out + x

        # feed forward sublayer 
        x = self.ln2(x)
        out = self.rff(x)
        x = out + x  
        x = self.ln2(x)
        
        if need_weights:
            return x, attn_weights
        else: 
            return x
    
    def norm_last_forward(self, x, need_weights):
        #  self attention sublayer  
        out = self.mha(x, x, x, need_weights=need_weights)
        if need_weights:
            out, attn_weights = out
        else:
            if type(out) == tuple:
                out = out[0]
        x = out + x
        x = self.ln1(x)

        # feed forward sublayer 
        out = self.rff(x)
        x = out + x  
        x = self.ln2(x)
        
        if need_weights:
            return x, attn_weights
        else: 
            return x

    def forward(self, X, need_weights=True):
        if self.norm_first:
            return self.norm_first_forward(X, need_weights)
        else:
            return self.norm_last_forward(X, need_weights)

class Encoder(nn.Module):
    def __init__(self, num_layers, num_features, num_heads, dim_ff, norm_first=False):
        super().__init__()
        self.encoder_kwargs = {'num_features':num_features, "num_heads":num_heads, "dim_ff":dim_ff, "norm_first":norm_first}
        layers = [EncoderLayer(**self.encoder_kwargs) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.class_token = nn.Parameter(0.1*torch.randn(num_features))
    
    def forward(self, orig_x, need_weights=True):
        # append the class token to the start of the input set 
        #batch_size = len(x)
        #class_token = self.class_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        #x = torch.cat([class_token, x], dim=1)
        x = torch.transpose(orig_x,1,0)
        batch_size = x.shape[1]
        class_token = self.class_token.unsqueeze(0).unsqueeze(0).expand(-1,batch_size, -1)
        x = torch.cat([class_token, x], dim=0)
        # forward pass
        out = x
        attn_weights = []
        for layer in self.layers:
            out = layer(out)
            if need_weights:
                out, _attn_weights = out
                attn_weights.append(_attn_weights)
            else:
                if type(out) == tuple:
                    out = out[0]

        out=torch.transpose(out,1,0)
        if need_weights: 
            return out, attn_weights 
        else:
            return out

class Activation(nn.Module):
    def __init__(self, fn,):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs)

class SetTransformerClassifier(nn.Module):
    def __init__(self, num_layers, num_features, num_heads, dim_ff, norm_first=False, clf_hidden_dims=[5,], clf_activation=F.relu, num_classes=2, clf_dropout=0.,L1_reg=0.0):
        super().__init__()
        # set up the classifier activation function 
        assert callable(clf_activation)
        if callable(clf_activation) and (not isinstance(clf_activation, nn.Module)):
            clf_activation = Activation(clf_activation,)

        # set up the encoder 
        self.encoder_kwargs = {'num_layers':num_layers, 'num_features':num_features, 'num_heads':num_heads, 'dim_ff':dim_ff, 'norm_first':norm_first}
        self.encoder = Encoder(**self.encoder_kwargs)
        if norm_first:
            self.encoding_ln = nn.LayerNorm(num_features)
        else:
            self.encoding_ln = None
        
        # set up the final classifier 
        features = [num_features,] + clf_hidden_dims
        clf = []
        for (in_features, out_features) in zip(features[:-1], features[1:]):
            clf.append(nn.Linear(in_features, out_features))
            clf.append(deepcopy(clf_activation))
            if clf_dropout > 0.:
                clf.append(nn.Dropout(clf_dropout))
        clf.append(nn.Linear(clf_hidden_dims[-1], num_classes))
        self.clf = nn.Sequential(*clf)
        
    
    def encode(self, x, need_weights=True):
        out = self.encoder(x, need_weights)
        if need_weights:
            out, attn_weights = out
        else:
            if type(out) == tuple:
                out = out[0]
        
        z = out[:,0] # the first element which corresponds to the classification token 
        if self.encoding_ln is not None:
            z = self.encoding_ln(z)
        
        if need_weights:
            return z, attn_weights 
        else:
            return z, None 
    
    def forward(self, x, need_weights=False):
        # encoding 
        embout,attn_weights  = self.encode(x, need_weights) 
        # classifier 
        #out = self.clf(embout) 
        out = torch.nn.functional.log_softmax(self.clf(embout), dim=-1)
        return out #, embout, attn_weights 