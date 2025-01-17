import json
import os
import sys

import gpt
import numpy as np
import qcd_ml
import qcd_ml.compat.gpt
import torch
from scipy.sparse.linalg import LinearOperator, eigs

torch.manual_seed(42)

# test vectors
test_vs = [
    torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble) for _ in range(10)
]

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
nr_layers = parameters["nr_layers"]
filter_iteration = parameters["filter_iteration"]
lens_space = parameters["lens_space"]
lens_time = parameters["lens_time"]
cheb_filter_iteration = parameters["cheb_filter_iteration"]

# load the gauge filed
loadpath = os.path.join(ensemble, ensemble_config)
U_gpt = gpt.load(loadpath)
U = [torch.tensor(qcd_ml.compat.gpt.lattice2ndarray(Umu)) for Umu in U_gpt]

# paths
paths = [[]]
for l in lens_space:
    paths.extend(
        [[(mu, l)] for mu in range(3)] + [[(mu, -l)] for mu in range(3)]
    )
for l in lens_time:
    paths.extend([[(3, l)], [(3, -l)]])


# create the model
class GENIE(torch.nn.Module):
    def __init__(self, U, paths):
        super(GENIE, self).__init__()

        self.U = U
        self.paths = paths
        self.pt = qcd_ml.nn.pt.v_PT(self.paths, self.U)

        self.dense_layers = torch.nn.ModuleList(
            [
                qcd_ml.nn.dense.v_Dense(1, len(self.paths)),
                *[
                    qcd_ml.nn.dense.v_Dense(len(self.paths), len(self.paths))
                    for _ in range(nr_layers - 1)
                ],
                qcd_ml.nn.dense.v_Dense(len(self.paths), 1),
            ]
        )

    def forward(self, v):
        v = torch.stack([v])
        for i in range(nr_layers):
            v = self.pt(self.dense_layers[i](v))
        v = self.dense_layers[-1](v)
        return v[0]


model = GENIE(U, paths)

# initialize weights
for li in model.dense_layers:
    li.weights.data = 0.001 * torch.randn_like(
        li.weights.data, dtype=torch.cdouble
    )
    li.weights.data[0, 0] += torch.eye(4)

# Wilson-clover Dirac operator
w = qcd_ml.qcd.dirac.dirac_wilson_clover(
    U, fermion_p["mass"], fermion_p["csw_r"]
)

optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

training_epochs = parameters["training_epochs"]
check_every = parameters["check_icg_every"]

cost = np.zeros(training_epochs)
its = np.zeros(training_epochs // check_every + 1)

# it_refs = np.zeros(len(test_vs))
# for i, test_v in enumerate(test_vs):
#    x, ret = qcd_ml.util.solver.GMRES(w, test_v, torch.zeros_like(test_v), eps=1e-8, maxiter=10000)
#    it_refs[i] = ret["k"]
# print(f"Reference Iteration count: {np.mean(it_refs)} +- {np.std(it_refs, ddof=1)/np.sqrt(len(test_vs))}")
#
# shift = -10
# l = 8 * 8 * 8 * 16 * 4 * 3
# w_np = (
#    lambda x: np.reshape(
#        qcd_ml.compat.gpt.lattice2ndarray(
#            w_gpt(
#                qcd_ml.compat.gpt.ndarray2lattice(
#                    np.reshape(x, (8, 8, 8, 16, 4, 3)),
#                    U_gpt[0].grid,
#                    gpt.vspincolor,
#                )
#            )
#        ),
#        (l,),
#    )
#    + shift * x
# )
# w_LinOp = LinearOperator((l, l), matvec=w_np)
# evs = eigs(w_LinOp, k=100, which='LM', return_eigenvectors=False, tol=1e-2)
# for ev in evs:
#    print((ev-shift).real, (ev-shift).imag)


# >>> Chebyshev stuff
def ChebApprox(a, b, n, func):
    c = np.zeros(n)
    bmah = 0.5 * (b - a)
    bpah = 0.5 * (b + a)
    fvals = np.zeros(n)
    for k in range(n):
        y = np.cos(np.pi * (k + 0.5) / n)
        fvals[k] = func(y * bmah + bpah)
    fac = 2.0 / n
    for j in range(n):
        s = 0.0
        for k in range(n):
            s += fvals[k] + np.cos(np.pi * j * (k + 0.5) / n)
        c[j] = fac * s
    c[0] /= 2
    return c


def ChebEval(c, m, A, v):
    d = torch.zeros_like(v)
    dd = torch.zeros_like(v)
    for j in range(m, 0, -1):
        sv = d
        d = 2 * A(d) - dd + c[j] * v
        dd = sv
    return A(d) - dd + c[0] * v


def w_trafo(v):
    return 2 * w(v) - (b + a) * v / (b - a)


# some magic numbers for Chebyshev approximation
# we approximate 1/x in [a, b] (smaller a leads to large fluctuations, b is set to roughly largest eigenvalues)
a = 0.5
b = 8
N = 1024  # how many Chebyshev polynomials to use to generate the coefficients (should be close to infinity)
c = ChebApprox(a, b, N, lambda x: 1 / x)


# <<< Chebyshev stuff

for t in range(training_epochs):
    tmps = [
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)
        for _ in range(batchsize)
    ]

    tmp2s = [
        qcd_ml.util.solver.GMRES(
            w,
            torch.zeros_like(tmp),
            tmp,
            eps=1e-8,
            maxiter=filter_iteration + 1,
        )[0]
        for tmp in tmps
    ]
    left_pres = [model.forward(w(tmp2)) for tmp2 in tmp2s]

    left_0s = [
        ChebEval(c, cheb_filter_iteration, w_trafo, left_pre)
        for left_pre in left_pres
    ]

    right_0s = [
        ChebEval(c, cheb_filter_iteration, w_trafo, tmp2) for tmp2 in tmp2s
    ]
    ins = left_0s
    outs = right_0s

    # normalize
    norms = [innerproduct(e, e) for e in ins]
    ins = [e / n**0.5 for e, n in zip(ins, norms)]
    outs = [e / n**0.5 for e, n in zip(outs, norms)]

    curr_cost = 0
    for ei, eo in zip(ins, outs):
        err = ei - eo
        curr_cost += innerproduct(err, err).real
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
        np.savetxt(
            os.path.join(out_path, f"residuals_{taskid}_sample{i}.dat"),
            ret_p["history"],
        )
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
