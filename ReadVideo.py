from decord import VideoReader
from decord import cpu, gpu


def ReadVideo(path):
    vr = VideoReader(path, ctx=cpu(0))
    # print('video frames:', len(vr))

    return vr