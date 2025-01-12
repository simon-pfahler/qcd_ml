import json
import os
import sys

import gpt
import numpy as np
import qcd_ml
import qcd_ml.compat.gpt
import torch
from qcd_ml.nn.dense import v_Add, v_Copy, v_Dense
from qcd_ml.nn.pt import v_PT
from scipy.sparse.linalg import LinearOperator, eigs

torch.manual_seed(42)

innerproduct = lambda x, y: (x.conj() * y).sum()

taskid = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open(f"../parameters/parameters.{taskid}.json", "r") as fin:
    parameters = json.load(fin)

# load parameters
ensemble = parameters["ensemble"]
ensemble_config = parameters["config"]
fermion_p = parameters["fermion_parameters"]
configno = ensemble_config.split(".")[0]
out_path = parameters["out_path"]
adam_lr = parameters["adam_lr"]
batchsize = parameters["batchsize"]

# load the gauge filed
loadpath = os.path.join(ensemble, ensemble_config)
U_gpt = gpt.load(loadpath)
U = [torch.tensor(qcd_ml.compat.gpt.lattice2ndarray(Umu)) for Umu in U_gpt]

# paths
paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
paths += [[(mu, 2)] for mu in range(4)] + [[(mu, -2)] for mu in range(4)]
paths += [[(mu, 4)] for mu in range(4)] + [[(mu, -4)] for mu in range(4)]
paths += [[(3, 8)], [(3, -8)]]


# create the model
class Deep_Model(torch.nn.Module):
    def __init__(self, U, paths):
        super(Deep_Model, self).__init__()

        self.U = U
        self.paths = paths
        self.layers = torch.nn.ModuleList(
            [
                v_Copy(len(self.paths)),
                v_PT(self.paths, self.U),
                v_Dense(len(self.paths), len(self.paths)),
                v_PT(self.paths, self.U),
                v_Dense(len(self.paths), len(self.paths)),
                v_PT(self.paths, self.U),
                v_Dense(len(self.paths), len(self.paths)),
                v_PT(self.paths, self.U),
                v_Dense(len(self.paths), len(self.paths)),
                v_PT(self.paths, self.U),
                v_Dense(len(self.paths), len(self.paths)),
                v_PT(self.paths, self.U),
                v_Dense(len(self.paths), len(self.paths)),
                v_PT(self.paths, self.U),
                v_Dense(len(self.paths), len(self.paths)),
                v_PT(self.paths, self.U),
                v_Add(len(self.paths)),
            ]
        )

    def forward(self, v):
        v = torch.stack([v])
        for layer in self.layers:
            v = layer(v)
        return v[0]


model = Deep_Model(U, paths)

# helper objects for weight initialization
idty = torch.eye(4)
zeros = torch.zeros((4, 4))

# Wilson-clover Dirac operator
# w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, fermion_p["mass"], fermion_p["csw_r"])
w_gpt = gpt.qcd.fermion.wilson_clover(U_gpt, fermion_p)
w = lambda x: torch.tensor(
    qcd_ml.compat.gpt.lattice2ndarray(
        w_gpt(
            qcd_ml.compat.gpt.ndarray2lattice(
                x.numpy(), U_gpt[0].grid, gpt.vspincolor
            )
        )
    )
)


# function to calculate mse
def complex_mse(output, target):
    err = output - target
    return (err.conj() * err).real.sum()


# function for l2norm
def l2norm(v):
    return (v.conj() * v).real.sum()


optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

training_epochs = parameters["training_epochs"]
check_every = parameters["check_icg_every"]

cost = np.zeros(training_epochs)
its = np.zeros(training_epochs // check_every + 1)

# test vector
test_vs = [
    torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble) for _ in range(10)
]
# it_refs = np.zeros(len(test_vs))
# for i, test_v in enumerate(test_vs):
#    x, ret = qcd_ml.util.solver.GMRES(w, test_v, torch.zeros_like(test_v), eps=1e-4, maxiter=10000)
#    it_refs[i] = ret["k"]
# print(f"Reference Iteration count: {np.mean(it_refs)} +- {np.std(it_refs, ddof=1)/np.sqrt(len(test_vs))}")

shift = -10
l = 8 * 8 * 8 * 16 * 4 * 3
w_np = (
    lambda x: np.reshape(
        qcd_ml.compat.gpt.lattice2ndarray(
            w_gpt(
                qcd_ml.compat.gpt.ndarray2lattice(
                    np.reshape(x, (8, 8, 8, 16, 4, 3)),
                    U_gpt[0].grid,
                    gpt.vspincolor,
                )
            )
        ),
        (l,),
    )
    + shift * x
)
w_LinOp = LinearOperator((l, l), matvec=w_np)
# evs = eigs(w_LinOp, k=100, which='LM', return_eigenvectors=False, tol=1e-2)
# for ev in evs:
#    print((ev-shift).real, (ev-shift).imag)

for t in range(training_epochs):
    v1s = [
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)
        for _ in range(batchsize)
    ]
    rv2s = [
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)
        for _ in range(batchsize)
    ]

    wv1s = [w(v1) for v1 in v1s]

    v2s = [
        qcd_ml.util.solver.GMRES(
            w, rv2, torch.zeros_like(rv2), eps=1e-8, maxiter=10
        )[0]
        for rv2 in rv2s
    ]
    wv2s = [w(v2) for v2 in v2s]

    ins = [*wv1s, *wv2s]
    outs = [*v1s, *v2s]

    # normalize
    norms = [l2norm(e) for e in ins]
    ins = [e / n**0.5 for e, n in zip(ins, norms)]
    outs = [e / n**0.5 for e, n in zip(outs, norms)]

    curr_cost = 0
    for ei, eo in zip(ins, outs):
        curr_cost += complex_mse(model.forward(ei), eo)
    curr_cost /= 8 * 8 * 8 * 16
    curr_cost /= batchsize

    cost[t] = curr_cost.item()
    optimizer.zero_grad()
    curr_cost.backward()
    optimizer.step()
    print(f"{t} - {cost[t]}")

    # if t % check_every == 0:
    #    with torch.no_grad():
    #        its = np.zeros(len(test_vs))
    #        for i, test_v in enumerate(test_vs):
    #            x_p, ret_p = qcd_ml.util.solver.GMRES(w, test_v, torch.zeros_like(test_v), preconditioner=lambda v: model.forward(v), eps=1e-8, maxiter=10000)
    #            its[i] = ret_p["k"]
    #        print(f"Model Iteration count: {np.mean(its)} +- {np.std(its, ddof=1)/np.sqrt(len(test_vs))}")

with torch.no_grad():
    its = np.zeros(len(test_vs))
    for i, test_v in enumerate(test_vs):
        x_p, ret_p = qcd_ml.util.solver.GMRES(
            w,
            test_v,
            torch.zeros_like(test_v),
            preconditioner=lambda v: model.forward(v),
            eps=1e-8,
            maxiter=10000,
        )
        its[i] = ret_p["k"]
print(
    f"Model Iteration count: {np.mean(its)} +- {np.std(its, ddof=1)/np.sqrt(len(test_vs))}"
)

shift = -10
l = 8 * 8 * 8 * 16 * 4 * 3


def mw_np(x):
    inp = torch.tensor(np.reshape(x, (8, 8, 8, 16, 4, 3)))
    with torch.no_grad():
        res = model.forward(w(inp)) + shift * inp
    return np.reshape(res.detach().numpy(), (8 * 8 * 8 * 16 * 4 * 3))


w_LinOp = LinearOperator((l, l), matvec=mw_np)
evs = eigs(w_LinOp, k=100, which="LM", return_eigenvectors=False, tol=1e-2)
for ev in evs:
    print((ev - shift).real, (ev - shift).imag)

# save weights
torch.save(
    list(model.parameters()),
    os.path.join(out_path, f"weights_1hxl_ptc_{configno}_{taskid}.pt"),
)

print("Done")
