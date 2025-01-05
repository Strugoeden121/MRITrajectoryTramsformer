import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=55555, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--data-path', type=pathlib.Path,
                      default='/home/tamir.shor/MRI/aug', help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.,
                      help='Fraction of total volumes to include')
    parser.add_argument('--test-name', type=str, default='test/', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='output/',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--ocmr-path',type=str,default='/home/tamir.shor/MRI/ocmr_data_attributes.csv', help="Path to ocmr attributes csv")
    parser.add_argument('--fsim',action='store_true',help="calculate fsim values (advised to only use this over a trained model, not in training - computing fsim takes ~30 secs)")
    parser.add_argument('--vif', action='store_true',
                        help="calculate vif values (advised to only use this over a trained model, not in training - computing vif takes ~30 secs)")

    # model parameters
    parser.add_argument('--num-layers', type=int, default=1, help='Number of VST Block layers')
    parser.add_argument('--depth', type=int, default=1, help='Depth of VST Block layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=96, help='Number of channels for feature extraction')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--decimation-rate', default=10, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')

    # optimization parameters
    parser.add_argument('--batch-size', default=12, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of training epochs per frame')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for reconstruction model')
    parser.add_argument('--lr-step-size', type=int, default=300,
                        help='Period of learning rate decay for reconstruction model')
    parser.add_argument('--lr-gamma', type=float, default=1,
                        help='Multiplicative factor of reconstruction model learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    # trajectory learning parameters
    parser.add_argument('--sub-lr', type=float, default=0.05, help='learning rate of the sub-samping layer')
    parser.add_argument('--sub-lr-time', type=float, default=300,
                        help='learning rate decay timestep of the sub-sampling layer')
    parser.add_argument('--sub-lr-stepsize', type=float, default=1,
                        help='learning rate decay step size of the sub-sampling layer')

    parser.add_argument('--trajectory-learning', default=False, action = "store_true",
                        help='trajectory_learning, if set to False, fixed trajectory, only reconstruction learning.')

    #MRI Machine Parameters
    parser.add_argument('--acc-weight', type=float, default=1e-2, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=1e-1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')
    parser.add_argument('--gamma', type=float, default=42576, help='gyro magnetic ratio - kHz/T')
    parser.add_argument('--G-max', type=float, default=40, help='maximum gradient (peak current) - mT/m')
    parser.add_argument('--S-max', type=float, default=180, help='maximum slew-rate - T/m/s')
    parser.add_argument('--FOV', type=float, default=0.2, help='Field Of View - in m')
    parser.add_argument('--dt', type=float, default=1e-5, help='sampling time - sec')
    parser.add_argument('--a-max', type=float, default=0.17, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=3.4, help='maximum velocity')
    parser.add_argument('--initialization', type=str, default='radial',
                        help='Trajectory initialization when using PILOT (spiral, EPI, rosette, uniform, gaussian).')
    parser.add_argument('--SNR', action='store_true', default=False,
                        help='add SNR decay')
    #modelization parameters
    parser.add_argument('--n-shots', type=int, default=16,
                        help='Number of shots')
    parser.add_argument('--interp_gap', type=int, default=10,
                        help='number of interpolated points between 2 parameter points in the trajectory')
    parser.add_argument('--num_frames_per_example', type=int, default=8, help='num frames per example')
    parser.add_argument('--boost', action='store_true', default=False, help='boost to equalize num examples per file')

    parser.add_argument('--project', action='store_true', default=True, help='Use projection to impose kinematic constraints.'
                                                                             'If false, use interpolation and penalty (original PILOT paper).')
    parser.add_argument('--proj_iters', default=10e1, help='Number of iterations for each projection run.')
    parser.add_argument('--multi_traj', action='store_true', default=True, help='allow different trajectory per frame')
    parser.add_argument('--augment', action='store_true', default=True, help='Use augmented files. data-path argument from should lead to augmented '
                                                                             'files generated by the augment script. If false,'
                                                                             'path should lead to relevant ocmr dataset')
    parser.add_argument('--noise', action='store_true', default=False, help='add noise to traj.')
    parser.add_argument('--traj-dropout', action='store_true', default=False, help='randomly fix traj coordinates.')
    parser.add_argument('--recons_resets', action='store_true', default=False, help='Use Reconstruction Resets.')
    parser.add_argument('--traj_freeze', action='store_true', default=False, help='Use Trajectory Freezing.')
    return parser


