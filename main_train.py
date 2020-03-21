from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--spec_norm',dest = "spec_mode", action="store_true")
    parser.add_argument('--overwrite',dest = "overwrite", action="store_true")
    parser.set_defaults(spec_mode=False)
    parser.set_defaults(overwrite=False)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if opt.overwrite == False:
        if (os.path.exists(dir2save)):
            print('trained model already exist')
        else:
            try:
                os.makedirs(dir2save)
            except OSError:
                pass
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            train(opt, Gs, Zs, reals, NoiseAmp)
            SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
