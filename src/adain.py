def calc_mean_std(feat, eps=1e-5):
    # feat shape: [batch, channels, height, width]
    size = feat.size()
    assert len(size) == 4

    batch_size, channels = size[:2]

    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2)
    feat_mean = feat_mean.view(batch_size, channels, 1, 1)

    feat_var = feat.view(batch_size, channels, -1).var(
        dim=2, unbiased=False
    ) + eps

    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)

    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()

    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_content = (
        content_feat - content_mean.expand(size)
    ) / content_std.expand(size)

    return normalized_content * style_std.expand(size) + style_mean.expand(size)
