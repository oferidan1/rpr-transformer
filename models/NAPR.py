import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from .backbone import build_backbone
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .transformer_encoder import TransformerEncoderLayer, TransformerEncoder,_get_clones, _get_activation_fn
from .transformer import TransformerDecoderLayer, TransformerDecoder

# positional encoding from nerf
def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


# PoseEncoder implementation from: https://github.com/yolish/camera-pose-auto-encoders/blob/main/models/pose_encoder.py
class PoseEncoder(nn.Module):

    def __init__(self, encoder_dim, apply_positional_encoding=True,
                 num_encoding_functions=6, shallow_mlp=False):

        super(PoseEncoder, self).__init__()
        self.apply_positional_encoding = apply_positional_encoding
        self.num_encoding_functions = num_encoding_functions
        self.include_input = True
        self.log_sampling = True
        x_dim = 3
        q_dim = 4
        if self.apply_positional_encoding:
            x_dim = x_dim + self.num_encoding_functions * x_dim * 2
            q_dim = q_dim + self.num_encoding_functions * q_dim * 2
        if shallow_mlp:
            self.x_encoder = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(),
                                           nn.Linear(64,encoder_dim))
            self.q_encoder = nn.Sequential(nn.Linear(q_dim, 64), nn.ReLU(),
                                           nn.Linear(64,encoder_dim))
        else:
            self.x_encoder = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(),
                                           nn.Linear(64,128),
                                           nn.ReLU(),
                                           nn.Linear(128,256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim)
                                           )
            self.q_encoder = nn.Sequential(nn.Linear(q_dim, 64), nn.ReLU(),
                                           nn.Linear(64, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim)
                                       )



        self.x_dim = x_dim
        self.q_dim = q_dim
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, pose):
        if self.apply_positional_encoding:
            encoded_x = positional_encoding(pose[:, :3])
            encoded_q = positional_encoding(pose[:, 3:])
        else:
            encoded_x = pose[:, :3]
            encoded_q = pose[:, 3:]

        latent_x = self.x_encoder(encoded_x)
        latent_q = self.q_encoder(encoded_q)
        return latent_x, latent_q


def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)
    return q

def compute_abs_pose_torch(rel_pose, abs_pose_neighbor):
    abs_pose_query = torch.zeros_like(rel_pose)
    abs_pose_query[:, :3] = abs_pose_neighbor[:, :3] + rel_pose[:, :3]
    abs_pose_query[:, 3:] = qmult(abs_pose_neighbor[:, 3:], rel_pose[:, 3:])
    return abs_pose_query


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)

class NAPR(nn.Module):
    def __init__(self, config, backbone_path):
        super().__init__()
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
        backbone_type = config.get("rpr_backbone_type")
        if backbone_type == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path))
            self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone_dim = 2048

        elif backbone_type == "resnet34":
            backbone = torchvision.models.resnet34(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))
            self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone_dim = 512

        elif backbone_type == "mobilenet":
            backbone = torchvision.models.mobilenet_v2(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 1280

        elif backbone_type == "efficientnet":
            # Efficient net
            self.backbone = torch.load(backbone_path)
            backbone_dim = 1280
        else:
            raise NotImplementedError("backbone type: {} not supported".format(backbone_type))
        self.backbone_type = backbone_type

        # CNN backbone
        self.backbone = build_backbone(config, backbone_path)

        # Encoders
        self.rpr_encoder_dim = config.get("rpr_hidden_dim")
        self.rpr_decoder_dim = config.get("rpr_hidden_dim")
        rpr_num_heads = config.get("rpr_num_heads")
        rpr_dim_feedforward = config.get("rpr_dim_feedforward")
        rpr_dropout = config.get("rpr_dropout")
        rpr_activation = config.get("rpr_activation")
        self.num_neighbors = config.get("num_neighbors")
        num_encoder_layers = config.get("rpr_num_encoder_layers")
        num_decoder_layers = config.get("rpr_num_decoder_layers")

        self.reductions = config.get("rpr_reduction")
        self.reduction_map = {"reduction_3": 40, "reduction_4": 112}
        self.backbone_num_channels = [self.reduction_map[reduction] for reduction in self.reductions]
        self.input_proj_x = nn.Conv2d(self.backbone.num_channels[0], self.rpr_decoder_dim, kernel_size=1)
        self.input_proj_q = nn.Conv2d(self.backbone.num_channels[1], self.rpr_decoder_dim, kernel_size=1)
        self.proj_ref = nn.Linear(backbone_dim, self.rpr_decoder_dim)

        encoder_layer = TransformerEncoderLayer(self.rpr_encoder_dim, rpr_num_heads, rpr_dim_feedforward,
                                                rpr_dropout, rpr_activation, True)
        encoder_norm = nn.LayerNorm(self.rpr_encoder_dim)
        self.rpr_transformer_encoder_x = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.rpr_transformer_encoder_q = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # transformer_encoder_layer = nn.TransformerEncoderLayer(
        #                                                        #d_model=rpr_encoder_dim_x,
        #                                                        d_model=self.rpr_encoder_dim,
        #                                                        nhead=rpr_num_heads,
        #                                                        dim_feedforward=rpr_dim_feedforward,
        #                                                        dropout=rpr_dropout,
        #                                                        activation=rpr_activation)
        #
        # self.rpr_transformer_encoder_x = nn.TransformerEncoder(transformer_encoder_layer,
        #                                                      num_layers=config.get("rpr_num_encoder_layers"),
        #                                                      #norm=nn.LayerNorm(rpr_encoder_dim_x))
        #                                                      norm=nn.LayerNorm(self.rpr_encoder_dim))
        #
        # self.rpr_transformer_encoder_q = nn.TransformerEncoder(transformer_encoder_layer,
        #                                                        num_layers=config.get("rpr_num_encoder_layers"),
        #                                                        #norm=nn.LayerNorm(rpr_encoder_dim_q))
        #                                                        norm=nn.LayerNorm(self.rpr_encoder_dim))

        decoder_layer = TransformerDecoderLayer(self.rpr_decoder_dim, rpr_num_heads, rpr_dim_feedforward,
                                                rpr_dropout, rpr_activation, True)
        decoder_norm = nn.LayerNorm(self.rpr_decoder_dim)
        self.rpr_transformer_decoder_x = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.rpr_transformer_decoder_q = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.rpr_decoder_dim, nhead=rpr_num_heads,
        #                                                        dim_feedforward=rpr_dim_feedforward,
        #                                                        dropout=rpr_dropout, activation=rpr_activation, #batch_first=True,
        #                                                        norm_first=True
        #                                                        )
        #
        # self.rpr_transformer_decoder_x = nn.TransformerDecoder(transformer_decoder_layer,
        #                                                  num_layers=config.get("rpr_num_decoder_layers"),
        #                                                  norm=nn.LayerNorm(self.rpr_decoder_dim))
        #
        # self.rpr_transformer_decoder_q = nn.TransformerDecoder(transformer_decoder_layer,
        #                                                    num_layers=config.get("rpr_num_decoder_layers"),
        #                                                    norm=nn.LayerNorm(self.rpr_decoder_dim))

        self.ln = nn.LayerNorm(self.rpr_decoder_dim)
        self.rel_regressor_x = PoseRegressor(self.rpr_decoder_dim, 3)
        self.rel_regressor_q = PoseRegressor(self.rpr_decoder_dim, 4)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward_backbone(self, img):
        '''
        Returns a global latent encoding of the img
        :param img: N x Cin x H x W
        :return: z: N X Cout
        '''
        if self.backbone_type == "efficientnet":
            z = self.backbone.extract_features(img)
        else:
            z = self.backbone(img)
        z = self.avg_pooling_2d(z).flatten(start_dim=1)
        return z

    def forward_backbone_dense(self, img):
        xs = self.backbone.extract_endpoints(img)
        out1 = xs[self.reductions[0]]
        out2 = xs[self.reductions[1]]
        #out1 = out1.flatten(start_dim=1)
        #out2 = out2.flatten(start_dim=1)
        return out1, out2

    def forward(self, data, encode_refs=True):
        '''
        :param query: N x Cin x H x W
        :param refs: N x K x Cin x H x W
        :param ref_pose: N x 7
        :param encode_refs: boolean, whether to encode the ref images
        :return: p the pose of the query
        '''
        query = data.get('query')
        refs = data.get('knn')

        # query image: Extract the features and the position embedding from the visual backbone middle dense layers
        #query_x, query_q = self.forward_backbone_dense(query)

        # Handle data structures
        if isinstance(query, (list, torch.Tensor)):
            query = nested_tensor_from_tensor_list(query)

        # Extract the features and the position embedding from the visual backbone
        features, pos = self.backbone(query)

        src_x, mask_x = features[0].decompose()
        src_q, mask_q = features[1].decompose()

        # Run through the transformer to translate to "camera-pose" language
        assert mask_x is not None
        assert mask_q is not None

        if encode_refs:
            n, k, h, w, c = refs.shape
            refs = refs.reshape(n*k, h, w, c)
            z_refs = self.backbone.forward_backbone(refs)
            z_refs = z_refs.reshape(n, k, -1)
        else:
            z_refs = refs

        # Prepare sequence (learned_token + neighbors)
        # todo consider giving different neighbors for x and q estimation
        z_refs = z_refs.transpose(0,1) # shape: K x  N x Cout
        bs = z_refs.shape[1]

        query_x = self.input_proj_x(src_x).flatten(2).permute(2, 0, 1)
        query_q = self.input_proj_q(src_q).flatten(2).permute(2, 0, 1)
        mask_x = mask_x.flatten(1)
        mask_q = mask_q.flatten(1)
        pos[0] = pos[0].flatten(2).permute(2, 0, 1)
        pos[1] = pos[1].flatten(2).permute(2, 0, 1)
        # pass position and orientation dense features through each encoder
        z_x = self.rpr_transformer_encoder_x(query_x, src_key_padding_mask=mask_x, pos=pos[0])
        z_q = self.rpr_transformer_encoder_q(query_q, src_key_padding_mask=mask_q, pos=pos[1])

        z_refs = self.proj_ref(z_refs)
        # apply decoder
        out_x = self.ln(self.rpr_transformer_decoder_x(z_refs, z_x, memory_key_padding_mask=mask_x, pos=pos[0])).squeeze().permute(1, 0, 2)
        out_q = self.ln(self.rpr_transformer_decoder_q(z_refs, z_q, memory_key_padding_mask=mask_q, pos=pos[1])).squeeze().permute(1, 0, 2) #.transpose(1, 2)
        # apply regressors
        returned_value = {}
        for i in range(self.num_neighbors):
            rel_x = self.rel_regressor_x(out_x[:, i, :])
            rel_q = self.rel_regressor_q(out_q[:, i, :])
            returned_value["rel_pose_{}".format(i)] = torch.cat((rel_x, rel_q), dim=1)

        return returned_value

'''
class NSRPR(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.get("input_dim")
        d_model = config.get("d_model")
        nhead = config.get("nhead")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout")
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer,
                                                               num_layers=config.get("num_decoder_layers"),
                                                               norm=nn.LayerNorm(d_model))
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(input_dim, d_model)
        self.cls1 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.cls2 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.rel_regressor_x = PoseRegressor(d_model, 3)
        self.rel_regressor_q = PoseRegressor(d_model, 4)
        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
    def forward(self, query, knn):
        knn = self.proj(knn)
        query = self.proj(query)
        # apply first classifier on knn
        knn_distr_before = self.cls1(knn)
        # apply decoder
        out = self.ln(self.transformer_decoder(knn, query.unsqueeze(1)))
        # apply second classifier on decoders outputs
        knn_distr_after = self.cls2(out)
        # apply regressors
        returned_value = {}
        num_neighbors = knn.shape[1]
        for i in range(num_neighbors):
            rel_x = self.rel_regressor_x(out[:, i, :])
            rel_q = self.rel_regressor_q(out[:, i, :])
            returned_value["rel_pose_{}".format(i)] = torch.cat((rel_x, rel_q), dim=1)
        returned_value["knn_distr_before"] = knn_distr_before
        returned_value["knn_distr_after"] = knn_distr_after
        # return the relative poses and the log-softmax from the first and second classifier
        return returned_value
class NS2RPR(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.get("input_dim")
        d_model = config.get("d_model")
        nhead = config.get("nhead")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout")
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_decoder_rot = nn.TransformerDecoder(transformer_decoder_layer,
                                                               num_layers=config.get("num_decoder_layers"),
                                                               norm=nn.LayerNorm(d_model))
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(input_dim, d_model)
        self.cls1 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.cls2 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.rel_regressor_x = PoseRegressor(d_model, 3)
        self.rel_regressor_q = PoseRegressor(d_model, 4)
        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
    def forward(self, query, knn):
        knn = self.proj(knn)
        query = self.proj(query)
        # apply first classifier on knn
        knn_distr_before = self.cls1(knn)
        # apply decoder
        out = self.ln(self.transformer_decoder(knn, query.unsqueeze(1)))
        # apply second classifier on decoders outputs
        knn_distr_after = self.cls2(out)
        # apply regressors
        returned_value = {}
        num_neighbors = knn.shape[1]
        for i in range(num_neighbors):
            rel_x = self.rel_regressor_x(out[:, i, :])
            rel_q = self.rel_regressor_q(out[:, i, :])
            returned_value["rel_pose_{}".format(i)] = torch.cat((rel_x, rel_q), dim=1)
        returned_value["knn_distr_before"] = knn_distr_before
        returned_value["knn_distr_after"] = knn_distr_after
        # return the relative poses and the log-softmax from the first and second classifier
        return returned_value
'''