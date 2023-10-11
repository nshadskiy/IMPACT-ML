import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

class LinearVariational(nn.Module):
    """
    Mean field approximation of nn.Linear
    """
    def __init__(self, in_features, out_features, parent, n_batches, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias        
        self.parent = parent
        self.n_batches = n_batches
        
        if getattr(parent, 'accumulated_kl_div', None) is None:
            parent.accumulated_kl_div = 0
            
        # Initialize the variational parameters.
        # Q(w)=N(mu_theta,sigma2_theta)
        # Do some random initialization with sigma=0.001
        self.w_mu = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=0, std=0.1)
        )
        # proxy for variance
        # log(1 + exp(ρ))* eps
        self.w_p = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=-3., std=0.1)
        )
        # # Prior mean
        # self.w_mu_pr = nn.Parameter(
        #     torch.FloatTensor(in_features, out_features).normal_(mean=0, std=0.1)
        # )

        if self.include_bias:
            self.b_mu = nn.Parameter(
                #torch.zeros(out_features)
                torch.FloatTensor(out_features).normal_(mean=0., std=0.1)
            )
            # proxy for variance
            self.b_p = nn.Parameter(
                #torch.zeros(out_features)
                torch.FloatTensor(out_features).normal_(mean=-3., std=0.1)
            )
            # # bias prior mean
            # self.b_mu_pr = nn.Parameter(
            #     #torch.zeros(out_features)
            #     torch.FloatTensor(out_features).normal_(mean=0., std=0.1)
            # )
        
    def reparameterize(self, mu, p):
        sigma = torch.log(1 + torch.exp(p)) 
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)
    
    def kl_divergence(self, mu_theta, p_theta, mu_prior, prior_sd=1.):
        # log_prior = dist.Normal(mu_prior, prior_sd).log_prob(z) 
        # log_p_q = dist.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z) 
        
        # return (log_p_q - log_prior).sum() / self.n_batches
        postr = dist.Normal(mu_theta, torch.log(1 + torch.exp(p_theta)))
        prior = dist.Normal(mu_prior, prior_sd)
        return dist.kl.kl_divergence(postr,prior).sum() / self.n_batches


    # def forward(self, x):
    #     w = self.reparameterize(self.w_mu, self.w_p)
        
    #     if self.include_bias:
    #         b = self.reparameterize(self.b_mu, self.b_p)
    #     else:
    #         b = 0
    
    #     z = x @ w + b

    #     z_mu, z_var = self.calcPredDist(x)
        
    #     self.parent.accumulated_kl_div += self.kl_divergence(w, self.w_mu, self.w_p, 0., 1.)
    #     if self.include_bias:
    #         self.parent.accumulated_kl_div += self.kl_divergence(b, self.b_mu, self.b_p, 0., 1.)
    #     return z, z_mu, z_var


    def forward(self, x):
        # sampling delta_W
        sigma_weight = torch.log1p(torch.exp(self.w_p))
        delta_weight = (sigma_weight * torch.randn_like(sigma_weight))

        if self.include_bias:
            sigma_bias = torch.log1p(torch.exp(self.b_p))
            bias = (sigma_bias * torch.randn_like(sigma_bias))
        else:
            bias = torch.zeros(self.out_features)

        # get kl divergence
        self.parent.accumulated_kl_div += self.kl_divergence(self.w_mu, self.w_p, 0., 1.)
        if self.include_bias:
            self.parent.accumulated_kl_div += self.kl_divergence(self.b_mu, self.b_p, 0., 1.)

        #print(self.w_mu,self.b_mu)
        #print(self.w_mu.size(),self.b_mu.size())
        # linear outputs
        outputs = F.linear(x, self.w_mu.transpose(0,1), self.b_mu)

        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        perturbed_outputs = F.linear(x * sign_input, delta_weight.transpose(0,1), bias) * sign_output

        z_mu, z_var = self.calcPredDist(x)

        # returning outputs + perturbations
        return outputs + perturbed_outputs, z_mu, z_var

    def calcPredDist(self, x):
        if self.include_bias:
            z_mu = x @ self.w_mu + self.b_mu
            #z_var = (x @ self.w_p)**2 + self.b_p**2
            z_var = x**2 @ torch.log(1 + torch.exp(self.w_p))**2 + torch.log(1 + torch.exp(self.b_p))**2
        else:
            z_mu = x @ self.w_mu
            z_var = x**2 @ torch.log(1 + torch.exp(self.w_p))**2
        return z_mu, z_var