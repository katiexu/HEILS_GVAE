import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from torchquantum.encoding import encoder_op_list_name_dict
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator

# PennyLane imports
import pennylane as qml
from Arguments import Arguments    # Only for setting qml.device()


def gen_arch(change_code, base_code):        # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    n_qubits = base_code[0]    
    arch_code = ([i for i in range(2, n_qubits+1, 1)] + [1]) * base_code[1]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code

def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:,0] - 1
        change_code = change_code.reshape(-1, length)    
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:            
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1,0)
            j += 1
    return single_dict

def translator(single_code, enta_code, trainable, arch_code, fold=1):
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code) 

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # number of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits]-1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits])-1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design

def single_enta_to_design(single, enta, arch_code, fold=1):
    """
    Generate a design list usable by QNET from single and enta codes

    Args:
        single: Single-qubit gate encoding, format: [[qubit, gate_config_layer0, gate_config_layer1, ...], ...]
                Each two bits of gate_config represent a layer: 00=Identity, 01=U3, 10=data, 11=data+U3
        enta: Two-qubit gate encoding, format: [[qubit, target_layer0, target_layer1, ...], ...]
              Each value represents the target qubit position in that layer
        arch_code_fold: [n_qubits, n_layers]

    Returns:
        design: List containing quantum circuit design info, each element is (gate_type, [wire_indices], layer)
    """
    design = []
    single = qubit_fold(single, 0, fold)
    enta = qubit_fold(enta, 1, fold)

    n_qubits, n_layers = arch_code

    # Process each layer
    for layer in range(n_layers):
        # First process single-qubit gates
        for qubit_config in single:
            qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The config for each layer is at position: 1 + layer*2 and 1 + layer*2 + 1
            config_start_idx = 1 + layer * 2
            if config_start_idx + 1 < len(qubit_config):
                gate_config = f"{qubit_config[config_start_idx]}{qubit_config[config_start_idx + 1]}"

                if gate_config == '01':  # U3
                    design.append(('U3', [qubit], layer))
                elif gate_config == '10':  # data
                    design.append(('data', [qubit], layer))
                elif gate_config == '11':  # data+U3
                    design.append(('data', [qubit], layer))
                    design.append(('U3', [qubit], layer))
                # 00 (Identity) skip

        # Then process two-qubit gates
        for qubit_config in enta:
            control_qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The target qubit position in the list: 1 + layer
            target_idx = 1 + layer
            if target_idx < len(qubit_config):
                target_qubit = qubit_config[target_idx] - 1  # Convert to 0-based index

                # If control and target qubits are different, add C(U3) gate
                if control_qubit != target_qubit:
                    design.append(('C(U3)', [control_qubit, target_qubit], layer))
                # If same, skip (equivalent to Identity)

    return design

def cir_to_matrix(x, y, arch_code, fold=1):
    # x = qubit_fold(x, 0, fold)
    # y = qubit_fold(y, 1, fold)

    qubits = int(arch_code[0] / fold)
    layers = arch_code[1]
    entangle = gen_arch(y, [qubits, layers])
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1,0)
    single = np.ones((qubits, 2*layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]    
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]
    arch = np.insert(single, [(2 * i) for i in range(1, layers+1)], entangle, axis=1)
    return arch.transpose(1, 0)

def shift_ith_element_right(original_list, i):
    """
    对列表中每个item的第i个元素进行循环右移一位
    
    Args:
        original_list: 原始列表，如 [[3, 0, 5], [4, 3, 6], [5, 1, 7], [1, 2, 8]]
        i: 要循环右移的元素索引，如 i=1 表示第二个元素
   
    """   
    ith_elements = [item[i] for item in original_list]    
    # 循环右移一位：最后一个元素移到开头
    shifted_ith = [ith_elements[-1]] + ith_elements[:-1]    
    result = [item[:i] + [shifted_ith[idx]] + item[i+1:] for idx, item in enumerate(original_list)]
    return result

def qubit_fold(jobs, phase, fold=1):
    if fold > 1:
        job_list = []
        for job in jobs:            
            if phase == 0:
                q = job[0]
                job_list += [[fold*(q-1)+1+i] + job[1:] for i in range(0, fold)]
            else:
                job = [i-1 for i in job]
                q = job[0]
                indices = [i for i, x in enumerate(job) if x < q]
                enta = [[fold*j+i+1 for j in job] for i in range(0,fold)]
                for i in indices:
                    enta = shift_ith_element_right(enta, i)
                job_list += enta
    else:
        job_list = jobs
    return job_list

class TQLayer_old(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        
        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(10)]

        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        # self.design['change_qubit'] = 3
        self.q_params_rot, self.q_params_enta = [], []
        for i in range(self.args.n_qubits):
            self.q_params_rot.append(pi * torch.rand(self.design['n_layers'], 3)) # each U3 gate needs 3 parameters
            self.q_params_enta.append(pi * torch.rand(self.design['n_layers'], 3)) # each CU3 gate needs 3 parameters
        rot_trainable = True
        enta_trainable = True

        for layer in range(self.design['n_layers']):
            for q in range(self.n_wires):

                # single-qubit parametric gates
                if self.design['rot' + str(layer) + str(q)] == 'U3':
                     self.rots.append(tq.U3(has_params=True, trainable=rot_trainable,
                                           init_params=self.q_params_rot[q][layer]))
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'CU3':
                    self.entas.append(tq.CU3(has_params=True, trainable=enta_trainable,
                                             init_params=self.q_params_enta[q][layer]))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def data_uploading(self, qubit):
        input = [      
        {"input_idx": [0], "func": "ry", "wires": [qubit]},        
        {"input_idx": [1], "func": "rz", "wires": [qubit]},        
        {"input_idx": [2], "func": "rx", "wires": [qubit]},        
        {"input_idx": [3], "func": "ry", "wires": [qubit]},  
        ]
        return input

    def forward(self, x, n_qubits=4, task_name=None):
        bsz = x.shape[0]
        if task_name.startswith('QML'):
            x = x.view(bsz, n_qubits, -1)
        else:
            kernel_size = self.args.kernel
            x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1,2)
            else:
                x = x.view(bsz, 4, 4).transpose(1,2)


        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
       

        for layer in range(self.design['n_layers']):            
            for j in range(self.n_wires):
                if self.design['qubit_{}'.format(j)][0][layer] != 0:
                    self.uploading[j](qdev, x[:,j])
                if self.design['qubit_{}'.format(j)][1][layer] == 0:
                    self.rots[j + layer * self.n_wires](qdev, wires=j)

            for j in range(self.n_wires):
                if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
                    self.entas[j + layer * self.n_wires](qdev, wires=self.design['enta' + str(layer) + str(j)][1])
        out = self.measure(qdev)
        if task_name.startswith('QML'):
            out = out[:, :2]    # only take the first two measurements for binary classification

        return out


class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits        
        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(self.n_wires)]

        self.q_params_rot = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each U3 gate needs 3 parameters
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each CU3 gate needs 3 parameters
        
        self.measure = tq.MeasureAll(tq.PauliZ)

    def data_uploading(self, qubit):
        input = [
            {"input_idx": [0], "func": "ry", "wires": [qubit]},
            {"input_idx": [1], "func": "rz", "wires": [qubit]},
            {"input_idx": [2], "func": "rx", "wires": [qubit]},
            {"input_idx": [3], "func": "ry", "wires": [qubit]},
        ]
        return input

    def forward(self, x):
        bsz = x.shape[0]
        kernel_size = self.args.kernel
        task_name = self.args.task        
        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        
        for i in range(len(self.design)):
            if self.design[i][0] == 'U3':                
                layer = self.design[i][2]
                qubit = self.design[i][1][0]
                params = self.q_params_rot[layer][qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.u3(qdev, wires=self.design[i][1], params=params)
            elif self.design[i][0] == 'C(U3)':               
                layer = self.design[i][2]
                control_qubit = self.design[i][1][0]
                params = self.q_params_enta[layer][control_qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.cu3(qdev, wires=self.design[i][1], params=params)
            else:   # data uploading: if self.design[i][0] == 'data'
                j = int(self.design[i][1][0])
                self.uploading[j](qdev, x[:,j])
        out = self.measure(qdev)
        if task_name.startswith('QML'):            
            out = out[:, :2]    # only take the first two measurements for binary classification            
        return out


class QiskitLayer(nn.Module):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.num_classes = len(self.args.digits_of_interest)

        # Trainable quantum circuit parameters
        self.u3_params = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3), requires_grad=True)  # Each U3 gate needs 3 parameters.
        self.cu3_params = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3), requires_grad=True) # Each C(U3) gate nees 3 parameters.

        # Setup Qiskit noise backend
        self.setup_qiskit_noise_backend()

    def setup_qiskit_noise_backend(self):
        """Setup Qiskit noise backend using GenericBackendV2"""
        try:
            # Use default settings to create GenericBackendV2, including basis_gates and coupling_map
            self.backend = GenericBackendV2(num_qubits=self.args.n_qubits)

            # Build noise model from backend properties
            self.noise_model = NoiseModel.from_backend(self.backend)
            print(f"✅ Successfully created noise model from Qiskit GenericBackendV2")
            print(f"   Using default basis gates: {self.backend.operation_names}")
        except Exception as e:
            print(f"❌ Error loading noise model from Qiskit GenericBackendV2: {e}")


    def create_quantum_circuit(self, x):
        # Preprocess data: downsample and flatten
        bsz = x.shape[0]
        kernel_size = self.args.kernel
        task_name = self.args.task
        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)

        quantum_circuits = []
        for batch in range(bsz):
            qc = QuantumCircuit(self.args.n_qubits)

            for i in range(len(self.design)):
                if self.design[i][0] == 'U3':
                    layer = self.design[i][2]
                    qubit = self.design[i][1][0]
                    theta = float(self.u3_params[layer, qubit, 0])
                    phi = float(self.u3_params[layer, qubit, 1])
                    lam = float(self.u3_params[layer, qubit, 2])
                    qc.u(theta, phi, lam, qubit)
                elif self.design[i][0] == 'C(U3)':
                    layer = self.design[i][2]
                    control_qubit = self.design[i][1][0]
                    target_qubit = self.design[i][1][1]
                    theta = float(self.cu3_params[layer, control_qubit, 0])
                    phi = float(self.cu3_params[layer, control_qubit, 1])
                    lam = float(self.cu3_params[layer, control_qubit, 2])
                    qc.cu(theta, phi, lam, 0, control_qubit, target_qubit)
                else:  # data uploading: if self.design[i][0] == 'data'
                    j = int(self.design[i][1][0])
                    qc.ry(float(x[batch][:, j][0].detach()), j)
                    qc.rx(float(x[batch][:, j][1].detach()), j)
                    qc.rz(float(x[batch][:, j][2].detach()), j)
                    qc.ry(float(x[batch][:, j][3].detach()), j)

            quantum_circuits.append(qc)

        return quantum_circuits

    def create_pauli_observables(self, physical_qubit_indices):
        """
        Create Pauli-Z observables based on physical qubit mapping
        physical_qubit_indices = [0, 1, 3, 2] means:
            - Logical qubit 0 maps to physical qubit 0 -> 'ZIII'
            - Logical qubit 1 maps to physical qubit 1 -> 'IZII'
            - Logical qubit 2 maps to physical qubit 3 -> 'IIIZ'
            - Logical qubit 3 maps to physical qubit 2 -> 'IIZI'
        """
        observables = []

        # Create observable for each physical qubit
        for i, physical_qubit_idx in enumerate(physical_qubit_indices):
            pauli_str = physical_qubit_idx * 'I' + 'Z' + (len(physical_qubit_indices) - 1 - physical_qubit_idx) * 'I'
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)

        return observables

    def run_qiskit_simulator(self, quantum_circuits, is_training=True):
        backend_seeds = 170
        # algorithm_globals.random_seed = backend_seeds
        seed_transpiler = backend_seeds
        shot = 6000

        # Decide whether to apply noise based on training or inference phase
        if is_training:
            use_noise = self.args.use_noise_model_train
            phase = "training"
        else:
            use_noise = self.args.use_noise_model_inference
            phase = "inference"

        if not hasattr(self, f'_printed_{phase}'):
            print(f"Running quantum simulation for {phase} phase - Noise: {use_noise}")
            setattr(self, f'_printed_{phase}', True)

        if use_noise:
            estimator = Estimator(
                backend_options={
                    'method': 'statevector',
                    'device': self.args.backend_device,
                    'noise_model': self.noise_model  # Add noise model when noise is enabled
                },
                run_options={
                    'shots': shot,
                    'seed': backend_seeds,
                },
                transpile_options={
                    'seed_transpiler': seed_transpiler
                }
            )
        else:
            estimator = Estimator(
                backend_options={
                    'method': 'statevector',
                    'device': self.args.backend_device,
                    # Do not use noise model when noise is disabled
                },
                run_options={
                    'shots': shot,
                    'seed': backend_seeds,
                },
                transpile_options={
                    'seed_transpiler': seed_transpiler
                }
            )

        results = []
        for i, qc in enumerate(quantum_circuits):
            transpiled_qc = transpile(qc, backend=self.backend)

            physical_qubit_indices = []
            for q in range(transpiled_qc.num_qubits):
                try:
                    initial_layout = str(transpiled_qc.layout.initial_layout[q])
                    index = int(initial_layout.split(', ')[-1].rstrip(')'))
                    physical_qubit_indices.append(index)
                except (KeyError, IndexError, ValueError, AttributeError) as e:
                    print(f"Warning: Could not extract mapping for physical qubit {q}: {e}")

            # Create Pauli-Z observables based on transpilation mapping (physical_qubit_indices)
            observables = self.create_pauli_observables(physical_qubit_indices)

            # Measure expectation values for each observable
            expectation_values = []
            for observable in observables:
                try:
                    job = estimator.run(transpiled_qc, observable)
                    result = job.result()
                    expectation_value = result.values[0]
                    expectation_values.append(expectation_value)
                except Exception as e:
                    print(f"Error running quantum circuit {i} for observable: {e}")
                    expectation_values.append(0.0)  # Default value when error occurs

            # Convert expectation values to tensor
            quantum_output = torch.tensor([expectation_values], dtype=torch.float32)
            results.append(quantum_output)

        # Stack results into shape [batch_size, num_classes]
        if results:
            quantum_results = torch.cat(results, dim=0)
        else:
            # Create default output if no results
            quantum_results = torch.zeros((len(quantum_circuits), self.num_classes), dtype=torch.float32)

        return quantum_results


    def forward(self, x):
        device = x.device

        # Create quantum circuits
        quantum_circuits = self.create_quantum_circuit(x)

        # Run qiskit simulator with phase information
        quantum_results = self.run_qiskit_simulator(quantum_circuits, is_training=self.training)
        quantum_results = quantum_results.to(device)

        # Ensure results require gradients
        if not quantum_results.requires_grad:
            quantum_results.requires_grad_(True)

        output = quantum_results

        return output


dev = qml.device("lightning.qubit", wires=Arguments().n_qubits)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_net(self, x):
    kernel_size = self.args.kernel
    task_name = self.args.task
    if not task_name.startswith('QML'):
        x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
        if kernel_size == 4:
            # x = x.view(bsz, 6, 6)
            # tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
            # x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            pass
        else:
            # x = x.view(bsz, 4, 4).transpose(1, 2)
            x = x.view(4, 4).transpose(0, 1)
    else:
        # x = x.view(bsz, self.n_wires, -1)
        pass

    for i in range(len(self.design)):
        if self.design[i][0] == 'U3':
            layer = self.design[i][2]
            qubit = self.design[i][1][0]
            phi = self.u3_params[layer, qubit, 0]
            theta = self.u3_params[layer, qubit, 1]
            omega = self.u3_params[layer, qubit, 2]
            qml.Rot(phi, theta, omega, wires=qubit)
        elif self.design[i][0] == 'C(U3)':
            layer = self.design[i][2]
            control_qubit = self.design[i][1][0]
            target_qubit = self.design[i][1][1]
            phi = self.cu3_params[layer, control_qubit, 0]
            theta = self.cu3_params[layer, control_qubit, 1]
            omega = self.cu3_params[layer, control_qubit, 2]
            qml.CRot(phi, theta, omega, wires=[control_qubit, target_qubit])
        else:  # data uploading: if self.design[i][0] == 'data'
            j = int(self.design[i][1][0])
            qml.RY(x[:, j][0].detach(), wires=j)
            qml.RX(x[:, j][1].detach(), wires=j)
            qml.RZ(x[:, j][2].detach(), wires=j)
            qml.RY(x[:, j][3].detach(), wires=j)

    return [qml.expval(qml.PauliZ(i)) for i in range(self.args.n_qubits)]

class PennylaneLayer(nn.Module):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.u3_params = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3), requires_grad=True)  # Each U3 gate needs 3 parameters
        self.cu3_params = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3), requires_grad=True) # Each CU3 gate needs 3 parameters

    def forward(self, x):
        output_list = []
        for batch in range(x.size(0)):  # Use actual batch size
            x_batch = x[batch]
            output = quantum_net(self, x_batch)
            q_out = torch.stack([output[i] for i in range(len(output))]).float()
            output_list.append(q_out)
        outputs = torch.stack(output_list)

        return outputs



class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        if arguments.backend == 'tq':
            print("Run with TorchQuantum backend.")
            self.QuantumLayer = TQLayer(self.args, self.design)
        elif arguments.backend == 'qi':
            print("Run with Qiskit quantum backend.")
            self.QuantumLayer = QiskitLayer(self.args, self.design)
        else:   # PennyLane or others
            print("Run with PennyLane quantum backend or others.")
            self.QuantumLayer = PennylaneLayer(self.args, self.design)

    def forward(self, x_image, n_qubits, task_name):
        # exp_val = self.QuantumLayer(x_image, n_qubits, task_name)
        exp_val = self.QuantumLayer(x_image)
        output = F.log_softmax(exp_val, dim=1)        
        return output

