import numpy as np
import torch 
import torch.nn as nn 
import cvxpy as cp 

class Pareto:
    def __init__(self, config = None):
        self.step = 0
        self.config = config
        self.optim_niter = config.optim_niter
        self.max_norm = config.max_norm
        self.n_tasks = config.n_tasks
        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.config.n_tasks)
        self.prvs_alpha = np.ones(self.config.n_tasks, dtype=np.float32)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=200)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []

        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(self, losses, parameters):
        extra_outputs = dict()
        shape_grad = []
        
        if self.step == 0:
            self._init_optim_problem()

        if self.step % self.config.update_weights_every == 0:
            self.step += 1
            grads = {}
            for i, loss in enumerate(losses):
                g = list(
                    torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=True,
                        allow_unused=True
                    )
                )
                if i == 0:
                    shape_grad = [grad.size() for grad in g if grad != None]
                
                lst_grad = []

                for j in range(len(g)): ## Doi voi loss contrastive khong co gradient layer implicit
                    if g[j] is None:
                        lst_grad.append(torch.flatten(torch.zeros(shape_grad[j])).to(self.config.device))
                    else:
                        lst_grad.append(torch.flatten(g[j]))
                
                lst_grad = torch.cat(lst_grad)
                grads[i] = lst_grad

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
        else:
            self.step += 1
            alpha = self.prvs_alpha
        
        if self.config.model == 'v1':
            alpha = alpha / sum(alpha)
        return alpha


    def find_weighted_loss(self, losses, parameters):
        weighted_loss = self.get_weighted_loss(
            losses = losses,
            parameters = parameters
        )
        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)

        return weighted_loss
        
    