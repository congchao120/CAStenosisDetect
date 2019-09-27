import numpy as np

rca_sens=[0.6875, 0.7708, 0.8750, 0.4583]
rca_senss=[0.4615, 0.5833, 0.6701, 0.3494]
rca_specs=[0.7601, 0.7759, 0.7519, 0.8106]
rca_mse=[37.1020, 49.6452, 37.5338, 26.2663]
print('rca_sens: ' + str(np.mean(rca_sens)) + '  ' + str(np.std(rca_sens)))
print('rca_senss: ' + str(np.mean(rca_senss)) + '  ' + str(np.std(rca_senss)))
print('rca_specs: ' + str(np.mean(rca_specs)) + '  ' + str(np.std(rca_specs)))
print('rca_mse: ' + str(np.mean(rca_mse)) + '  ' + str(np.std(rca_mse)))

lca_sens=[0.5926, 0.6071, 0.5926, 0.8571, 0.8205, 0.6000, 0.7500, 0.7000, 0.6667, 0.6250, 0.5750, 0.7750]
lca_senss=[0.3243, 0.2987, 0.3559, 0.5526, 0.5333, 0.3545, 0.5413, 0.6374, 0.3608, 0.4235, 0.3246, 0.6042]
lca_specs=[0.7561, 0.7584, 0.7115, 0.6460, 0.6127, 0.5482, 0.6049, 0.5161, 0.8310, 0.6494, 0.8123, 0.7666]
lca_mse=[26.3227, 24.6150, 32.5799, 43.9960, 64.2385, 64.1887, 46.5366, 54.6380, 18.3566, 43.8613, 23.5817, 28.7878]
print('lca_sens: ' + str(np.mean(lca_sens)) + '  ' + str(np.std(lca_sens)))
print('lca_senss: ' + str(np.mean(lca_senss)) + '  ' + str(np.std(lca_senss)))
print('lca_specs: ' + str(np.mean(lca_specs)) + '  ' + str(np.std(lca_specs)))
print('lca_mse: ' + str(np.mean(lca_mse)) + '  ' + str(np.std(lca_mse)))



rca_cam_sens=[0.6460, 0.7577, 0.7819]
rca_cam_senss=[0.6460, 0.3205, 0.5622]
rca_cam_specs=[0.5328, 0.73, 0.80]
rca_cam_mse=[81.7762, 0.73, 0.80]
print('rca_cam_sens: ' + str(np.mean(rca_cam_sens)) + '  ' + str(np.std(rca_cam_sens)))
print('rca_cam_senss: ' + str(np.mean(rca_cam_senss)) + '  ' + str(np.std(rca_cam_senss)))
print('rca_cam_specs: ' + str(np.mean(rca_cam_specs)) + '  ' + str(np.std(rca_cam_specs)))
print('rca_cam_mse: ' + str(np.mean(rca_cam_mse)) + '  ' + str(np.std(rca_cam_mse)))

lca_cam_sens=[0.6460, 0.7577, 0.7819]
lca_cam_senss=[0.6460, 0.3205, 0.5622]
lca_cam_specs=[0.5328, 0.73, 0.80]
lca_cam_mse=[81.7762, 0.73, 0.80]
print('lca_cam_sens: ' + str(np.mean(lca_cam_sens)) + '  ' + str(np.std(lca_cam_sens)))
print('lca_cam_senss: ' + str(np.mean(lca_cam_senss)) + '  ' + str(np.std(lca_cam_senss)))
print('lca_cam_specs: ' + str(np.mean(lca_cam_specs)) + '  ' + str(np.std(lca_cam_specs)))
print('lca_cam_mse: ' + str(np.mean(lca_cam_mse)) + '  ' + str(np.std(lca_cam_mse)))