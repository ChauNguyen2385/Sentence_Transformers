import io
import os
import logging
import torch
import numpy as np

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


logger = logging.getLogger(__name__)

from Base_handler import BaseHandler
class Sentence_Transformer(BaseHandler):

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("../sentence-transformers-1/sentence_transformers")

        # Read model serialize/pt file
        model_pt_path = os.path.join('./artefacts', "model.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "SentenceTransformer.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        import sys
        sys.path.append("../sentence-transformers-1/sentence_transformers")
        import SentenceTransformer
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = SentenceTransformer("./new_v2_3/pytorch_model.bin")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        # """
        #  Scales, crops, and normalizes a PIL image for a MNIST model,
        #  returns an Numpy array
        # """
        # image = data[0].get("data")
        # if image is None:
        #     image = data[0].get("body")

        # mnist_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        # image = Image.open(io.BytesIO(image))
        # image = mnist_transform(image)
        # return image
        text1 = self.model.encode(data[0])
        text2 = self.model.encode(data[1])
        return [text1,text2]
    def inference(self,data):
        result = self.model.util.cos_sim(data[0],data[1])
        return [result]

    def postprocess(self, inference_output):
        return inference_output


_service = Sentence_Transformer()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data


# torch-model-archiver --model-name st --version 1.0 --model-file ../sentence-transformers-1/sentence_transformers/SentenceTransformer.py --serialized-file artefacts/model.pt --handler handler.py