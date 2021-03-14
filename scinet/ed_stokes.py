#   Copyright 2018 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import cPickle
import gzip
import io

def stokes_eqn(Fr, St, t, x0=0.0, v0=0.0):
    #solution to equation x'' + 1/St*x' + Fr = 0
    t = np.array(t)
    C1 = -Fr*St - v0
    C2 = x0 - C1*St
    return np.real( -Fr*St * t + C1*St * np.exp(-t/St) + C2 )

def stokes_data(N, t_sample=np.linspace(0, 5, 50), Fr_interval=[0, 5], St_interval=[-2,2],
                    t_meas_interval=None, fileName=None):

    t_sample = np.array(t_sample, dtype=float)

    # sample in intervals
    Fr_all = (Fr_interval[1] - Fr_interval[0]) * np.random.rand(N) + Fr_interval[0]
    St_log_all = (St_interval[1] - St_interval[0]) * np.random.rand(N) + St_interval[0]
    St_all = 10**St_log_all

    #If measurement time not provided, create it to extrapolate
    if t_meas_interval is None:
        t_meas_interval = [t_sample[0], 2 * t_sample[-1]]
    t_meas = np.reshape(np.random.rand(N) * (t_meas_interval[1] - t_meas_interval[0]) + t_meas_interval[0], [N, 1])

    x_in = []
    x_out = []

    for Fr, St, t in zip(Fr_all, St_all, t_meas):
        x_in.append(  stokes_eqn(Fr, St, t_sample ) ) 
        x_out.append( stokes_eqn(Fr, St, t        ) ) 

    x_in = np.array(x_in)
    x_out = np.reshape(x_out, [N, 1])
    state_list = np.vstack([Fr_all, St_all]).T
    result = ([x_in, t_meas, x_out], state_list, [])

    if fileName is not None:
        f = gzip.open(io.data_path + fileName + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return (result)
