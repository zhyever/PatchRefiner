import torch


class RandomBBoxQueries(object):
    def __init__(self, batch_size, h, w, window_sizes, N=100):
        b = batch_size
        self.h, self.w = h, w
        queries = {}
        self.window_sizes = window_sizes

        for win_size in window_sizes:
            # queries[win_size]
            k = win_size // 2
            x = torch.randint(k+1, w - k, (b, N, 1))
            y = torch.randint(k+1, h - k, (b, N, 1))
            queries[win_size] = torch.cat((x,y), dim=-1)

        self.absolute = queries
        self.normalized = self._normalized()

    def _normalized(self):
        """returns queries in -1,1 range"""
        normed = {}
        for win_size, coords in self.absolute.items():
            c = coords.clone().float()
            c[:,:,0] = c[:,:,0] / (self.w - 1)  # w - 1 because range is [-1,1]
            c[:,:,1] = c[:,:,1] / (self.h - 1)
            normed[win_size] = c
        return normed

    def to(self, device):
        for win_size in self.window_sizes:
            self.absolute[win_size] = self.absolute[win_size].to(device)
            self.normalized[win_size] = self.normalized[win_size].to(device)
        return self

    def __repr__(self):
        return str(self.normalized)