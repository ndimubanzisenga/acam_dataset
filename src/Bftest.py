import numpy as np
import acoular
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import time
from os import makedirs
from os.path import exists
from os import walk, makedirs

DEBUG_MODE = False
SAVE_RESULTS = True
VISUALIZE_RESULTS = True
result_dir = None
DB_RANGE = 15

TITLE = None

class Data():
    def __init__(self, data_file_name, img_file_name, sample_freq=50000):
        self.time_data = None
        self.num_data_observations = 0
        self.num_data_channels = 0
        self.sample_freq = sample_freq
        self.img_setup = cv.imread(img_file_name) #ToDo: point to video file instead of simple image
                
        def init_data_attributes(data_file_name):
            data = np.loadtxt(data_file_name, delimiter=',')
            data = data * (20 / (float(2**31 )))
            self.time_data = data
            self.num_data_observations, self.num_data_channels = data.shape
            
        init_data_attributes(data_file_name)



#ToDo: dictionary to select beamformer technique
class BfTest():
    def __init__(self, data, num_bf_data_scans, bf_block_size):
        self.data = data
        self.num_bf_data_scans = num_bf_data_scans
        self.num_bf_data_frames = data.num_data_observations / num_bf_data_scans
        self.bf_block_size = bf_block_size
        self.running_durations = list()
        self.result_dir = None
        
    
    def run_bf_test(self, freq, z, bf_technique):
        def init_time_sample_obj():
            time_sample.numsamples = self.num_bf_data_scans
            time_sample.numchannels = self.data.num_data_channels
            time_sample.sample_freq = self.data.sample_freq
        
        def init_bf_grid(z):
            foX = 65.24
            foY = 51.28
            x = z * np.tan(np.deg2rad(foX/2))
            y = z * np.tan(np.deg2rad(foY/2))
            inc = x /25.
            rg = acoular.RectGrid( x_min=-x, x_max=x, y_min=-y, y_max=y, z=z, increment=inc )
            return rg
            
        def compute_bf_map(ts):
            Lm = np.empty((0,0))
            #ind_low = 1
            #ind_high = 50
            c = 343. #346.04
            bf = None
            ps = acoular.EigSpectra(time_data=ts, window='Hanning', block_size=self.bf_block_size, cached=False)
            if bf_technique == 1:
                bf = acoular.BeamformerFunctional(freq_data=ps, grid=rect_grid, mpos=mic_geom, r_diag=True, c=c, gamma=4, cached=False, steer='true location')
            elif bf_technique == 2:
                bf = acoular.BeamformerBase( freq_data=ps, grid=rect_grid, mpos=mic_geom, r_diag=True, c=c, cached=False, steer='true location')
            elif bf_technique == 3:
                bf = acoular.BeamformerCleansc(freq_data=ps, grid=rect_grid, mpos=mic_geom, r_diag=True, c=c, cached=False, steer='true location')
            pm = bf.synthetic(freq, octave_band)
            Lm = acoular.L_p(pm)
            
            if (DEBUG_MODE):
                print("Computes the Bf Map for frequencies between: {0}").format(ps._get_freq_range())
                print("Bf map shape {0}").format(Lm.shape)
            return Lm

        def visualize_bf_map(Lm, count):
            Lm_ = Lm
            label = str(freq) + '_' + str(count)
            m = Lm.min()
            if m < 0:
                Lm = Lm + abs(m)
            img_map= cv.threshold(Lm, Lm.max()-10, Lm.max(), cv.THRESH_TOZERO)[1] * 255 / Lm.max()
            img_map_gray = np.array(img_map, dtype=np.uint8)
            img_map_gray = cv.resize(img_map_gray, (self.data.img_setup.T.shape[1:]))
            img_map_cm = cv.applyColorMap(img_map_gray, cv.COLORMAP_HOT)

            merged_image = cv.addWeighted(self.data.img_setup, 0.8, img_map_cm, 0.2, 0.0)
            
            cv.imshow("Test", merged_image)
            cv.waitKey(10)
            return

        time_sample = acoular.TimeSamples()
        init_time_sample_obj()
        mic_geom = acoular.MicGeom(from_file='./../../../data/mic_configurations/acam_array_40.xml')
        bf_grid_Z = z
        rect_grid = init_bf_grid(bf_grid_Z)
        octave_band = 3
        
        count = 0
        result_Lm = None
        result_grid = None
        for data_frame in np.array_split(self.data.time_data, self.num_bf_data_frames):
            start_time = time.time()
            time_sample.data = data_frame               
            Lm = compute_bf_map(time_sample)
            print("Max SPL : {0}  and Min SPL : {1}").format(Lm.max(), Lm.min())

            if VISUALIZE_RESULTS:
                #visualize_bf_map(Lm, count)
                pass

            if count == 0:
                result_Lm = Lm
                result_grid = rect_grid.extend()

            self.running_durations.append(time.time() - start_time)
            count = count + 1

            if (DEBUG_MODE):
                print("TS object traits:\n")
                time_sample.print_traits()
                print("\nGrid object traits:\n")
                rect_grid.print_traits()
                print("\n\n")
            
        return (result_Lm, result_grid)


def visualize_results(results, result_dir):
    #plt.subplots(nrows=len(results), ncols=4)
    Lm_MinMax_list = list()
    for (backgound_img, Lm_list, grid, stats) in results:
        min_Lm = 350
        max_Lm = -350
        for Lm in Lm_list:
            min_Lm = min(min_Lm, Lm.max()-DB_RANGE)
            max_Lm = max(max_Lm, Lm.max())
        Lm_MinMax_list.append((min_Lm, max_Lm))
        
    im2 = None
    i = 0
    fig, axes = plt.subplots(nrows=5, ncols=4)
    for (backgound_img, Lm_list, grid_list, stats) in results:
        j = 0
        for Lm in Lm_list:
            ax = axes[i][j]
            im1 = ax.imshow(backgound_img, interpolation='nearest', alpha=1.0, extent=grid_list[j])
            im2 = ax.imshow(cv.resize(Lm_list[j], (backgound_img.T.shape[1:])), vmin=Lm_MinMax_list[i][0],\
                     vmax=Lm_MinMax_list[i][1], interpolation='nearest', alpha=0.3, cmap=plt.cm.hot, extent=grid_list[j])
            ax.set_xticks(grid_list[j][:2])
            if i == 4:
                ax.set_xlabel('X axis [m]')
            if j == 0:
                ax.set_ylabel('Y axis [m]')
            j = j + 1
            
        CB = plt.colorbar(im2, ax=axes[i].ravel().tolist(), pad=0.05, extend='both')
        CB.ax.set_ylabel('SPL [dB]')
        i = i + 1

    # Plot and save statistics
    '''ii = 0
    for (backgound_img, Lm_list, grid_list, stats) in results:
        plt.figure(ii)
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Time [s]")
        plt.title(TITLE)
        plt.plot(stats)
        ii = ii + 1'''

    # Save results
    if SAVE_RESULTS:
        if not exists(result_dir):
            makedirs(result_dir)
        file_name = result_dir+'/'+'result'+'.png'
        fig.savefig(file_name)


def test_dataset(base_directory, bf_technique):
    #base_directory = '../../../../test_folder/leakage/'
    num_scans = 512 * 50
    block_size = 512
    freqs = [2500, 5000, 8000, 10000]
    d = [1.0, 1.5, 2.0, 2.5, 3.0]
    N_freqs = len(freqs)
    directories_list = list()
    results = list()
    TITLE = ("Times taken to compute BF Map for: {0} scans and {1} block_size").format(num_scans, block_size)
    
    for (path, dirs, files) in walk(base_directory):
        directories_list = dirs
        directories_list.sort()
        break
    
    count = 0
    for directory in directories_list:
        dataset_path = base_directory + directory
        
        data_file_name = dataset_path + '/MicRawSignals.txt'
        img_file_name = dataset_path + '/images/OpticalImage.png'
        
        data = Data(data_file_name=data_file_name, img_file_name=img_file_name)
        bf = BfTest(data, num_bf_data_scans=num_scans, bf_block_size=block_size)
        
        Lm_list = list()
        grid_list = list()
        running_durations = []
        for f in freqs:
            Lm, grid = bf.run_bf_test(f, d[count], bf_technique)
            Lm_list.append(Lm)
            grid_list.append(grid)
            running_durations = np.r_[running_durations, bf.running_durations]
            
        results.append((bf.data.img_setup, Lm_list, grid_list, running_durations))
        count = count + 1
        
    return results

def main():
    if len(sys.argv) < 1:
        print("Enter dataset name ..")
        return
    else:
        print("#### Dataset :{0} #####").format(sys.argv[1])

    base_directory = '../../../../test_folder/' + str(sys.argv[1]) + '/'
    bf_technique = int(sys.argv[2]) # 1: for Functional BF, and 2: for Conventional BF
    result_dir = '../../../docs/results/' + str(bf_technique) + '_' + str(sys.argv[1])

    results = test_dataset(base_directory, bf_technique)

    if VISUALIZE_RESULTS:
        visualize_results(results, result_dir)
        plt.show()

    return
    
if __name__ == "__main__":
    main()