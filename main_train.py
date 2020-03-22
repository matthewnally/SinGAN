from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--input_name2', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--spec_norm',dest = "spec_mode", action="store_true")
    parser.add_argument('--overwrite',dest = "overwrite", action="store_true")
    parser.set_defaults(spec_mode=True)
    parser.set_defaults(overwrite=False)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    # if (os.path.exists(dir2save)):
    #     print('trained model already exist')
    # else:
    #     try:
    #         os.makedirs(dir2save)
    #     except OSError:
    #         pass
    #     real = functions.read_image(opt)
    #     functions.adjust_scales2image(real, opt)
    #     train(opt, Gs, Zs, reals, NoiseAmp)
    #     SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)

    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    real_ = functions.read_images(opt)
    real = [imresize(real_[0],opt.scale1,opt), imresize(real_[1],opt.scale1,opt)]
    reals = [functions.creat_reals_pyramid(real[0],reals,opt), functions.creat_reals_pyramid(real[1],reals,opt)]
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt, gen_start_scale = 4)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt, gen_start_scale = 8)
