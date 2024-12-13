import cirq
import numpy as np
import cmath as m
from scipy.stats import unitary_group
import matplotlib.pyplot as plt


d=3                                                                                               #dimensions of qudit
rep = 1

i=m.sqrt(-1)
sim = cirq.DensityMatrixSimulator()
w = m.exp(i*2*m.pi/d)


class nDH(cirq.Gate):
    def _qid_shape_(self):
        return (d,)

    def _unitary_(self):
        tmp = np.zeros((d,d), dtype=np.csingle)
        for i in range(0,d):
            for j in range(0,d):
                tmp[i,j] = w**(i*j)/m.sqrt(d)
        return tmp
    
    
    def _circuit_diagram_info_(self, args):
        return f'H({d})'

class Cplus(cirq.Gate):
    def __init__(self):
        self.d = d
    def _num_qubits_(self):
        return 2
    def _qid_shape_(self):
        return (d,)*2
    def _unitary_(self):
        tmp = np.zeros((d*d,d*d), dtype=np.csingle)
        for i in range(0,d):
            for j in range(0,d):
                tmp[d*i+j,d*i+(-i+j)%d] = 1
        return tmp
    def _circuit_diagram_info_(self,args):
        return '@', f'C+({d})'

class Cpinv(cirq.Gate):
    def __init__(self):
        self.d = d
    def _num_qubits_(self):
        return 2
    def _qid_shape_(self):
        return (d,)*2
    def _unitary_(self):
        tmp = np.zeros((d*d,d*d), dtype=np.csingle)
        for i in range(0,d):
            for j in range(0,d):
                tmp[d*i+j,d*i+(i+j)%d] = 1
        return tmp
    def _circuit_diagram_info_(self,args):
        return '@', f'C-({d})'

class Weyl(cirq.Gate):
    def __init__(self,Msg, Al):
        self.d = d
        self.msg = Msg
        self.al = Al

    def _num_qubits_(self):
        return 1

    def _qid_shape_(self):
        return (d,)

    def _unitary_(self):
        bla = np.zeros((d,d), dtype=np.csingle)
        for k in range(0,d):
            bla[k,(self.al+k)%d] = w**(self.msg*k*(d-1))
        print(bla)
        return bla

    def _circuit_diagram_info_(self, args):
        return f"U({self.msg}{self.al})"

class unitary(cirq.Gate):
    def __init__(self):
        self.d = d
    def _num_qubits_(self):
        return 3
    def _qid_shape_(self):
        return (d,) * 3
    def _unitary_(self):
        tmp = np.zeros((d*d*d,d*d*d), dtype=np.csingle)
        for i in range(0,d):
            for j in range(0,d):
                for k in range(0,d):
                    tmp[d*d*i+d*j+k,d*d*i+d*j+(j+k)%d] =  w**(i*k*(d-1))
        return tmp
    def _circuit_diagram_info_(self,args):
        return '@','@', f'U({self.d})'

class rand(cirq.Gate):
    def _qid_shape_(self):
        return (d,)

    def _unitary_(self):
        return bla

    def _circuit_diagram_info_(self, args):
        return 'rand'

class raninv(cirq.Gate):
    def _qid_shape_(self):
        return (d,)

    def _unitary_(self):
        return bla.conj().T

    def _circuit_diagram_info_(self, args):
        return 'rinv'

count = np.zeros((d))

for r in range(rep):
    bla = unitary_group.rvs(d)
    circuit = cirq.Circuit()
    Message, Alice, Bob = cirq.LineQid.range(3, dimension=d)
    circuit.append(rand()(Message))                                                                         #applying random gate to the initial state to get the message
    circuit.append([nDH()(Alice),Cplus()(Alice,Bob)])                                                       #generating the entangled state
    circuit.append([Cpinv()(Message,Alice),nDH()(Message)])
    #circuit.append(unitary()(Message,Alice,Bob))                                                            #applying 3-qudit gate to Bob's instead of the controlled gates
    
    for p in range(0,d):
        for q in range(0,d):
            circuit.append(Weyl(p,q)(Bob).controlled_by(Message, Alice, control_values= [p, q]))            #applying the controlled gates
    
    circuit.append(raninv()(Bob))
    circuit.append(cirq.measure(Message))                                                                   #measurement on the system
    circuit.append(cirq.measure(Alice))
    circuit.append(cirq.measure(Bob))

    result = sim.run(circuit)
    count[result._records[f'q(2) (d={d})'][0][0][0]] = count[result._records[f'q(2) (d={d})'][0][0][0]]+1

print(circuit)                                                                                          #prints the circuit for the teleportation protocol

print(count)
plt.bar(np.arange(0,d), count, color='b', width=0.25, label = 'Bob')
plt.xlabel('States')
plt.ylabel('Counts')
plt.xticks([r for r in range(d)], np.arange(0,d))
plt.legend()
plt.show()