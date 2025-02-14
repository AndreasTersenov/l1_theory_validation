import numpy as np
import healpy as hp
import multiprocessing as mp
import sys

class TakahashiLoader:
    def __init__(self, data_path="/feynman/work/dap/lcs/vt272285/data/sim_data", takahashi_data_path=None):
        self.data_path = data_path
        self.takahashi_data_path = takahashi_data_path or data_path + "/Takahashi/"

    def get_beam(self, theta, lmax):
        def top_hat(b, radius):
            return np.where(abs(b) <= radius, 1 / (np.cos(radius) - 1) / (-2 * np.pi), 0)

        t = theta * np.pi / (60 * 180)
        b = np.linspace(0.0, t * 1.2, 10000)
        bw = top_hat(b, t)
        beam = hp.sphtfunc.beam2bl(bw, b, lmax)
        return beam

    def smooth_map(self, args):
        theta, lmax_takahashi, kappa_1024, nside_takahashi = args
        beam = self.get_beam(theta, lmax_takahashi)
        almkappa = hp.sphtfunc.map2alm(kappa_1024)
        kappa_smooth = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almkappa, beam), nside_takahashi)
        return kappa_smooth

    def load_takahashi_file(self, filename, theta, new_nside, nprocess):
        skip = [0, 536870908, 1073741818, 1610612728, 2147483638, 2684354547, 3221225457]
        load_blocks = [skip[i+1]-skip[i] for i in range(0, 6)]
        with open(filename, 'rb') as f:
            rec = np.fromfile(f, dtype='uint32', count=1)[0]
            nside = np.fromfile(f, dtype='int32', count=1)[0]
            npix = np.fromfile(f, dtype='int64', count=1)[0]
            rec = np.fromfile(f, dtype='uint32', count=1)[0]
            print("nside:{} npix:{}".format(nside, npix))

            rec = np.fromfile(f, dtype='uint32', count=1)[0]
            print("file opened!")
            kappa = np.array([])
            r = npix
            for i, l in enumerate(load_blocks):
                blocks = min(l, r)
                load = np.fromfile(f, dtype='float32', count=blocks)
                np.fromfile(f, dtype='uint32', count=2)
                kappa = np.append(kappa, load)
                r = r-blocks
                if r == 0:
                    break
                elif r > 0 and i == len(load_blocks)-1:
                    load = np.fromfile(f, dtype='float32', count=r)
                    np.fromfile(f, dtype='uint32', count=2)
                    kappa = np.append(kappa, load)
            # gamma1 = np.array([])
            # r = npix
            # print("loop1 done! ")
            # for i, l in enumerate(load_blocks):
            #     blocks = min(l, r)
            #     load = np.fromfile(f, dtype='float32', count=blocks)
            #     np.fromfile(f, dtype='uint32', count=2)
            #     gamma1 = np.append(gamma1, load)
            # gamma2 = np.array([])
            # r = npix
            # for i, l in enumerate(load_blocks):
            #     blocks = min(l, r)
            #     load = np.fromfile(f, dtype='float32', count=blocks)
            #     np.fromfile(f, dtype='uint32', count=2)
            #     gamma2 = np.append(gamma2, load)
            #     r = r-blocks
            #     if r == 0:
            #         break
            #     elif r > 0 and i == len(load_blocks)-1:
            #         load = np.fromfile(f, dtype='float32', count=r)
            #         np.fromfile(f, dtype='uint32', count=2)
            #         gamma2 = np.append(gamma2, load)
            # omega = np.array([])
            # r = npix
            # print("loop 3 done!")
            # for i, l in enumerate(load_blocks):
            #     blocks = min(l, r)
            #     load = np.fromfile(f, dtype='float32', count=blocks)
            #     np.fromfile(f, dtype='uint32', count=2)
            #     omega = np.append(omega, load)
            #     r = r-blocks
            #     if r == 0:
            #         break
            #     elif r > 0 and i == len(load_blocks)-1:
            #         load = np.fromfile(f, dtype='float32', count=r)
            #         np.fromfile(f, dtype='uint32', count=2)
            #         omega = np.append(omega, load)

        nside_takahashi = new_nside
        lmax_takahashi = nside_takahashi * 3 - 1

        kappa_1024 = hp.pixelfunc.ud_grade(kappa, nside_takahashi)

        # Setting up arguments for parallel smoothing
        args1 = (theta, lmax_takahashi, kappa_1024, nside_takahashi)
        args2 = (theta * 2, lmax_takahashi, kappa_1024, nside_takahashi)

        # # Parallel smoothing operation
        # with mp.Pool(processes=nprocess) as pool:
        #     kappa_smooth1, kappa_smooth2 = pool.map(self.smooth_map, [args1, args2])
        
        # Non-parallel smoothing operation
        kappa_smooth1 = self.smooth_map(args1)
        kappa_smooth2 = self.smooth_map(args2)

        print("Smoothing operations done!")
        return np.var(kappa_smooth2 - kappa_smooth1), kappa_smooth1, kappa_smooth2

    def run_loader(self, theta1, nside, zs, nprocess):
        tk_fn = np.array([1, 9, 16, 18, 20, 22, 24, 25, 27, 34, 38])
        tk_zs = np.array([0.0506, 0.5078, 1.0334, 1.2179, 1.4230, 1.6528, 1.9121, 2.0548, 2.3704, 3.9309, 5.3423])
        # zs = tk_zs[2]
        print("the source redshift for which the file is being loaded is: ", zs)
        index = np.where(zs == tk_zs)[0][0]
        takahashi_file = self.takahashi_data_path + "allskymap_nres12r000.zs" + str(tk_fn[index]) + ".mag.dat"
        print("the takahashi file being loaded is: ", takahashi_file)
        var, kmap1, kmap2 = self.load_takahashi_file(takahashi_file, theta1, nside, nprocess)
        print("The redshift is:", zs)
        print("The theta is:", theta1)
        print("Variance:", var)
        return var, kmap1, kmap2
    
    
    
    
    