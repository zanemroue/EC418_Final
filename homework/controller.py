import pystk
import numpy as np

def control(aim_point, current_vel, steer_gain=6.5, skid_thresh=0.25, target_vel=30):
    action = pystk.Action()

    action.steer = np.clip(aim_point[0] * steer_gain, -1, 1)

    speed_error = target_vel - current_vel
    speed_gain = 0.2
    action.acceleration = np.clip(speed_gain * speed_error, 0, 1)
    action.brake = speed_error < -5

    action.drift = abs(aim_point[0]) > skid_thresh

    action.nitro = abs(action.steer) < 0.4

    return action




if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)


#RESULTS:

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller zengarden -v         
# Finished at t=404
# 404 0.9981356022052131

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller lighthouse -v        
# Finished at t=437
# 437 0.9980535614331709

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller hacienda -v          
# Finished at t=585
# 585 0.9986835805979872

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller snowtuxpeak -v       
# Finished at t=568
# 568 0.999269536396353

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller cornfield_crossing -v
# Finished at t=672
# 672 0.9986430505236459

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller scotland -v 
# Finished at t=662
# 662 0.9988821620366168


