# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-03-12 17:28:33
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-03-14 12:55:17
import jax
from matplotlib.pyplot import *
import numpy as np
import pynapple as nap
from examples_utils import data
from scipy.ndimage import gaussian_filter
import sys, os
import nemos as nmo
from scipy.io import loadmat
import pandas as pd

# ################
# units = loadmat("/mnt/home/gviejo/Downloads/Achilles_10252013_spikes_cellinfo.mat")
# celltype = loadmat("/mnt/home/gviejo/Downloads/Achilles_10252013.CellClass.cellinfo.mat")
# unit_id = units['spikes'][0][0][1].flatten()
# spikes = {}
# for i, n in enumerate(unit_id):
# 	spikes[n] = nap.Ts(t=units['spikes'][0][0][2][0][i].flatten())
# spikes = nap.TsGroup(spikes, 
# 	shank = units['spikes'][0][0][3].flatten(), 
# 	location = np.array([units['spikes'][0][0][7][0][i][0] for i in range(len(unit_id))]),
# 	cell_type = np.array([celltype['CellClass'][0][0][3][0][i][0] for i in range(len(unit_id))])
# 	)
# position_info = loadmat("/mnt/home/gviejo/Downloads/position_info.mat", simplify_cells=True)
# position = nap.Tsd(t=position_info['pos_inf']['ts'], d=position_info['pos_inf']['lin_pos'], time_support = spikes.time_support)
# theta_info = loadmat("/mnt/home/gviejo/Downloads/theta.mat", simplify_cells=True)
# theta = nap.Tsd(t=theta_info['theta']['time'], d=theta_info['theta']['phase'])
# ################

spikes = nap.load_file(os.path.expanduser("~/Dropbox/Achilles_10252013/Achilles_10252013_spikes.npz"))
position = nap.load_file(os.path.expanduser("~/Dropbox/Achilles_10252013/Achilles_10252013_position.npz"))
theta = nap.load_file(os.path.expanduser("~/Dropbox/Achilles_10252013/Achilles_10252013_theta.npz"))


# only the pyr
spikes = spikes.getby_category("cell_type")['pE'].getby_threshold("rate", 0.1)

# position 
position = position.dropna(update_time_support=True)#.find_support(1.0)

# taking only the forward 
forward_ep = np.array([[s,e] for s,e in position.time_support.values if position.get(e) - position.get(s) > 0])
forward_ep = nap.IntervalSet(start=forward_ep[:,0], end=forward_ep[:,1])

position = position.restrict(forward_ep)


# Speed
speed = []
for s, e in position.time_support.values:
	speed.append(np.pad(np.abs(np.diff(position.get(s, e))), [0, 1], mode='edge')*position.rate)
speed = nap.Tsd(t=position.t, d=np.hstack(speed), time_support = position.time_support)


# Tuning curves
tc_pf = nap.compute_1d_tuning_curves(spikes, position, 50, position.time_support)
tc_sp = nap.compute_1d_tuning_curves(spikes, speed, 20, position.time_support)


############
bin_size = 0.005

phase = theta.restrict(position.time_support).bin_average(bin_size)
speed = speed.interpolate(phase, position.time_support)
position = position.interpolate(phase, position.time_support)
count = spikes.count(bin_size, position.time_support)

############
# NEMOS

position_basis = nmo.basis.MSplineBasis(n_basis_funcs=10)
phase_basis = nmo.basis.CyclicBSplineBasis(n_basis_funcs=12)
speed_basis = nmo.basis.MSplineBasis(n_basis_funcs=15)

basis = position_basis*phase_basis + speed_basis

X = basis.evaluate(position, phase, speed)

X = X[:,None,:]

neuron = 10

glm = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized())
glm.fit(X, count[:,[neuron]])


w_pos = glm.coef_[0, 0:10]
w_phase = glm.coef_[0,10:22]
w_speed = glm.coef_[0,-15:]


samples, eval_basis = speed_basis.evaluate_on_grid(100)

# basis2 = position_basis*phase_basis
# samples2, eval_basis2 = basis2.evaluate_on_grid(100)

lin_pred = np.exp(
        glm.intercept_ +
        np.dot(np.mean((position_basis*phase_basis).evaluate(position, phase), axis=0), 
        	glm.coef_[0, 0:120]) +
        np.dot(speed_basis.evaluate(speed), glm.coef_[0, -15:])
		)/bin_size

tc_sp_glm = nap.compute_1d_tuning_curves_continuous(lin_pred[:,None], speed, 20, position.time_support)

pr = glm.predict(X)/bin_size
tc_pf_glm = nap.compute_1d_tuning_curves_continuous(pr, position, 50, position.time_support)



figure()
subplot(221)
plot(np.dot(eval_basis, w_speed))
subplot(222)
plot(tc_sp.values[:,neuron])
plot(tc_sp_glm.values[:,0])
subplot(224)
plot(tc_pf.values[:,neuron])
plot(tc_pf_glm.values[:,0])
show()


figure()


plot(position, color = 'black')
[axvspan(s,e, alpha = 0.4) for s,e in position.time_support.values]

figure()
for i in range(10*11):
	subplot(10, 11, i+1)
	fill_between(tc_pf.index.values, np.zeros(len(tc_pf)), tc_pf.values[:,i])

figure()
for i in range(10*11):
	subplot(10, 11, i+1)
	fill_between(tc_sp.index.values, np.zeros(len(tc_sp)), tc_sp.values[:,i])
	title(i)

show()



