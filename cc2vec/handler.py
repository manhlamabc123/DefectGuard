import torch, os, pickle, json, logging
from model import HierachicalRNN, DeepJITExtended
import numpy as np

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure the file handler
file_handler = logging.FileHandler('cc2vec.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class CC2VecHandler:
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.cc2vec = None
        self.deepjit_extended = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # Set up model parameters
        dictionary = pickle.load(open(model_dir + "/qt_dict.pkl", 'rb'))   
        dict_msg, dict_code = dictionary

        with open(model_dir + "/cc2vec.json", 'r') as file:
            params = json.load(file)

        # Set up param
        params["filter_sizes"] = [int(k) for k in params["filter_sizes"].split(',')]
        params["vocab_msg"], params["vocab_code"] = len(dict_msg), len(dict_code)
        params["cc2vec_class_num"] = len(dict_msg)
        params["deepjit_class_num"] = 1
        params["embedding_feature"] = params['embedding_size'] * 3 + 2 + 2

        # Initialize model
        self.cc2vec = HierachicalRNN(params).to(device=self.device)
        self.cc2vec.load_state_dict(torch.load(model_pt_path, map_location=self.device))

        self.deepjit_extended = DeepJITExtended(params).to(device=self.device)
        self.deepjit_extended.load_state_dict(torch.load(model_dir + "/deepjit_extended.pt", map_location=self.device))

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        return preprocessed_data


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        # Extract data from DataLoader
        added_code = np.array(model_input["input"]["added_code"])
        removed_code = np.array(model_input["input"]["removed_code"])
        code = torch.tensor(model_input["input"]["code"], device=self.device)
        message = torch.tensor(model_input["input"]["message"], device=self.device)

        # CC2Vec Forward
        state_word = self.cc2vec.init_hidden_word(self.device)
        state_sent = self.cc2vec.init_hidden_sent(self.device)
        state_hunk = self.cc2vec.init_hidden_hunk(self.device)
        feature = self.cc2vec(added_code, removed_code, state_hunk, state_sent, state_word, self.device)

        # DeepJIT Forward
        predict = self.deepjit_extended.forward(feature, message, code)
        return predict

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output.item()
        return [postprocess_output]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)