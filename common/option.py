import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Optional, Any


class BaseOptions:
    """Base class for all options.
    
    This class handles the basic configuration options and provides common functionality
    for both training and testing options.
    """
    
    def __init__(self, debug: bool = False):
        self.initialized = False
        self.debug = debug
        self.opt: Optional[Any] = None
        self.parser: Optional[argparse.ArgumentParser] = None
        self.isTrain: bool = False

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Initialize the parser with basic options.
        
        Args:
            parser: The argument parser to initialize
            
        Returns:
            The initialized parser
        """
        # Model configuration
        parser.add_argument('--model', type=str, default='WVLUT',
                          help='Model architecture to use')
        parser.add_argument('--model_type', type=str, default='shared',
                          help='Model architecture type')
        parser.add_argument('--nf', type=int, default=64,
                          help='Number of filters in convolutional layers')
        parser.add_argument('--modes', type=str, default='cdy',
                          help='Sampling modes to use in every stage')
        parser.add_argument('--interval', type=int, default=4,
                          help='N bit uniform sampling')
        
        # Path configuration
        parser.add_argument('--modelRoot', type=str, default='../models',
                          help='Root directory for models')
        parser.add_argument('--expDir', '-e', type=str, default='',
                          help='Experiment directory')
        
        # Debug configuration
        parser.add_argument('--load_from_opt_file', action='store_true', default=False,
                          help='Load options from file')
        parser.add_argument('--debug', default=False, action='store_true',
                          help='Enable debug mode')

        self.initialized = True
        return parser

    def gather_options(self) -> argparse.Namespace:
        """Gather all options from command line and option file.
        
        Returns:
            The parsed options
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt = parser.parse_args("") if self.debug else parser.parse_args()

        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        self.parser = parser
        return opt

    def print_options(self, opt: argparse.Namespace) -> None:
        """Print options to console.
        
        Args:
            opt: The options to print
        """
        message = []
        message.append('----------------- Options ---------------')
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: {default}]'
            message.append(f'{str(k):>25}: {str(v):<30}{comment}')
        message.append('----------------- End -------------------')
        print('\n'.join(message))

    def save_options(self, opt: argparse.Namespace) -> None:
        """Save options to file.
        
        Args:
            opt: The options to save
        """
        file_name = os.path.join(opt.expDir, 'opt')
        
        # Save as text file
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = f'\t[default: {default}]'
                opt_file.write(f'{str(k):>25}: {str(v):<30}{comment}\n')

        # Save as pickle file
        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser: argparse.ArgumentParser, 
                               opt: argparse.Namespace) -> argparse.ArgumentParser:
        """Update parser with options from file.
        
        Args:
            parser: The parser to update
            opt: Current options
            
        Returns:
            Updated parser
        """
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt: argparse.Namespace) -> argparse.Namespace:
        """Load options from file.
        
        Args:
            opt: Current options
            
        Returns:
            Loaded options
        """
        file_name = self.option_file_path(opt, makedir=False)
        return pickle.load(open(file_name + '.pkl', 'rb'))

    def save_code(self) -> None:
        """Save current code to experiment directory."""
        src_dir = "./"
        trg_dir = os.path.join(self.opt.expDir, "code")
        for f in Path(src_dir).rglob("*.py"):
            trg_path = os.path.join(trg_dir, f)
            os.makedirs(os.path.dirname(trg_path), exist_ok=True)
            shutil.copy(os.path.join(src_dir, f), trg_path, follow_symlinks=False)

    def parse(self, save: bool = False) -> argparse.Namespace:
        """Parse options.
        
        Args:
            save: Whether to save options
            
        Returns:
            Parsed options
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # opt = self.process(opt)

        # Setup experiment directory
        if opt.expDir == '':
            opt.modelDir = os.path.join(opt.modelRoot, "debug")
            os.makedirs(opt.modelDir, exist_ok=True)

            count = 1
            while os.path.isdir(os.path.join(opt.modelDir, f'expr_{count}')):
                count += 1
            opt.expDir = os.path.join(opt.modelDir, f'expr_{count}')
            os.makedirs(opt.expDir)
        else:
            os.makedirs(opt.expDir, exist_ok=True)

        opt.modelPath = os.path.join(opt.expDir, "Model.pth")

        if opt.isTrain:
            opt.valoutDir = os.path.join(opt.expDir, 'val')
            os.makedirs(opt.valoutDir, exist_ok=True)
            self.save_options(opt)

        if opt.isTrain and opt.debug:
            opt.displayStep = 10
            opt.saveStep = 100
            opt.valStep = 50
            opt.totalIter = 200

        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):
    """Training options class."""
    
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Initialize training options.
        
        Args:
            parser: The parser to initialize
            
        Returns:
            Initialized parser
        """
        parser = super().initialize(parser)
        
        # Data configuration
        parser.add_argument('--batchSize', type=int, default=16,
                          help='Batch size for training')
        parser.add_argument('--cropSize', type=int, default=48,
                          help='Input LR training patch size')
        parser.add_argument('--trainDir', type=str, default="../data/LOL_v1",
                          help='Training data directory')
        parser.add_argument('--valDir', type=str, default='../data/LowLightBenchmark',
                          help='Validation data directory')
        parser.add_argument('--valDataset', type=str, nargs='+', default=["val"],
                          help='Validation dataset names')
        
        # Training configuration
        parser.add_argument('--startIter', type=int, default=0,
                          help='Starting iteration (0 for from scratch)')
        parser.add_argument('--totalIter', type=int, default=200,
                          help='Total number of training iterations')
        parser.add_argument('--displayStep', type=int, default=10,
                          help='Display info every N iteration')
        parser.add_argument('--valStep', type=int, default=50,
                          help='Validate every N iteration')
        parser.add_argument('--saveStep', type=int, default=50,
                          help='Save models every N iteration')
        
        # Optimization configuration
        parser.add_argument('--lr0', type=float, default=1e-3,
                          help='Initial learning rate')
        parser.add_argument('--lr1', type=float, default=1e-4,
                          help='Final learning rate')
        parser.add_argument('--weightDecay', type=float, default=0,
                          help='Weight decay')
        
        # Hardware configuration
        parser.add_argument('--gpuNum', '-g', type=int, default=1,
                          help='Number of GPUs to use')
        parser.add_argument('--workerNum', '-n', type=int, default=8,
                          help='Number of workers for data loading')

        self.isTrain = True
        return parser


class TestOptions(BaseOptions):
    """Testing options class."""
    
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Initialize testing options.
        
        Args:
            parser: The parser to initialize
            
        Returns:
            Initialized parser
        """
        parser = super().initialize(parser)

        parser.add_argument('--loadIter', '-i', type=int, default=200,
                          help='Iteration to load model from')
        parser.add_argument('--testDir', type=str, default='../data/LowLightBenchmark',
                          help='Test data directory')
        parser.add_argument('--valDataset', type=str, nargs='+', default=["val"],
                          help='Validation dataset names')
        parser.add_argument('--resultRoot', type=str, default='../results',
                          help='Root directory for results')
        parser.add_argument('--lutName', type=str, default='LUT_ft',
                          help='Name of the LUT to use')
        parser.add_argument('--ifsave', type=str, default='false',
                          help='Whether to save output images (true/false)')
        parser.add_argument('--ifGT', type=str, default='false',
                          help='Whether to use GT-based normalization (true/false)')
        parser.add_argument('--scale_show', type=str, default='false',
                          help='Whether to show scale information (true/false)')

        self.isTrain = False
        return parser
