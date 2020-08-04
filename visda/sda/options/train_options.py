from .base_options import BaseOptions
import os.path as osp


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://10.5.36.31", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='reid', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8088, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=25, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=25, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=20, help='multiply by a gamma every lr_decay_iters iterations')
        # data
        parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc')
        parser.add_argument('-dt', '--dataset-target', type=str, default='market1501')
        # parser.add_argument('-b', '--batch-size', type=int, default=64)
        parser.add_argument('-sb', '--sync-batch-size', type=int, default=16)
        # parser.add_argument('-j', '--workers', type=int, default=4)
        parser.add_argument('--num-clusters', type=int, default=500)
        parser.add_argument('--generations', type=int, default=1)
        parser.add_argument('--start-generation', type=int, default=0)
        parser.add_argument('--resume', type=str, default='')
        parser.add_argument('--train-mode', type=str, default='joint', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')


        parser.add_argument('--gan-pretrain-epochs', type=int, default=0)
        # parser.add_argument('--height', type=int, default=256,
        #                     help="input height")
        # parser.add_argument('--width', type=int, default=128,
        #                     help="input width")
        parser.add_argument('--num-instances', type=int, default=4,
                            help="each minibatch consist of "
                                 "(batch_size // num_instances) identities, and "
                                 "each identity has num_instances instances, "
                                 "default: 0 (NOT USE)")
        parser.add_argument('--source-erasing', action='store_true')

        # parser.add_argument('--lambda_tri', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
        # model
        parser.add_argument('-a', '--arch', type=str, default='resnet50')
        parser.add_argument('--features', type=int, default=0)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--dropout-tri', type=float, default=0)
        parser.add_argument('--alpha', type=float, default=0.999)
        parser.add_argument('--joint-train', action='store_true')
        # argsimizer
        parser.add_argument('--lr-reid', type=float, default=0.00035,
                            help="learning rate of new parameters, for pretrained "
                                 "parameters it is 10 times smaller than this")
        parser.add_argument('--momentum', type=float, default=0.9)
        # parser.add_argument('--alpha', type=float, default=0.999)
        # parser.add_argument('--moving-avg-momentum', type=float, default=0)
        parser.add_argument('--weight-decay', type=float, default=5e-4)
        # parser.add_argument('--soft-ce-weight', type=float, default=0.5)
        # parser.add_argument('--soft-tri-weight', type=float, default=0.5)
        parser.add_argument('--epochs', type=int, default=40)
        parser.add_argument('--iters', type=int, default=0)
        # training configs
        parser.add_argument('--init', type=str, default='', metavar='PATH')
        parser.add_argument('--init-s', type=str, default='', metavar='PATH')
        parser.add_argument('--init-t', type=str, default='', metavar='PATH')
        parser.add_argument('--relation-type', type=str, default='triplet')
        # parser.add_argument('--print-freq', type=int, default=1)
        parser.add_argument('--eval-step', type=int, default=1)
        parser.add_argument('--lambda-value', type=float, default=0.1)
        parser.add_argument('--cluster-step', type=int, default=1)
        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--tcp-port', default='2080', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-url', default='tcp://10.1.72.197', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--launcher', default='tcp', type=str)

        self.isTrain = True
        return parser
