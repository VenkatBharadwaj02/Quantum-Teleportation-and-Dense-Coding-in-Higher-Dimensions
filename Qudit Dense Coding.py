
import cirq
import numpy as np
import cmath as m

d=5
sim = cirq.Simulator()
i = m.sqrt(-1)
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
        return bla

    def _circuit_diagram_info_(self, args):
        return f"U({self.msg}{self.al})"


# Create a simulator
sim = cirq.Simulator()
circuit = cirq.Circuit()
Alice, Bob = cirq.LineQid.range(2, dimension=d)
circuit.append([nDH()(Alice),Cplus()(Alice,Bob)])
# Define the message you want to send
message = "01"

circuit.append(Weyl(int(message[0]),int(message[1]))(Alice))
    
circuit.append([Cpinv()(Alice,Bob),nDH()(Alice)])
circuit.append([cirq.measure(Alice), cirq.measure(Bob)])

# Print the circuit
print("Dense Coding Circuit:")
print(circuit)

# Simulate the circuit for 100 repetitions
result = sim.run(circuit, repetitions=100)
print(result)
