from torch.optim import lr_scheduler


def get_cyclical_lr_scheduler(optimizer,
                              base_lr,
                              max_lr,
                              step_size_down=None,
                              mode='exp_range',
                              gamma=0.98,
                              scale_fn=None,
                              scale_mode='cycle',
                              cycle_momentum=True,
                              base_momentum=0.8,
                              max_momentum=0.9,
                              last_epoch=-1):
    """
    cycle based scale_fn example
    scale_mode='cycle'
    clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

    iteration based scale_fn example
    scale_mode='iterations'
    clr_fn = lambda x: 1/(5**(x*0.0001))
    """

    return lr_scheduler.CyclicLR(optimizer,
                                 base_lr,
                                 max_lr,
                                 step_size_down=step_size_down,
                                 mode=mode,
                                 gamma=gamma,
                                 scale_fn=scale_fn,
                                 scale_mode=scale_mode,
                                 cycle_momentum=cycle_momentum,
                                 base_momentum=base_momentum,
                                 max_momentum=max_momentum,
                                 last_epoch=last_epoch)
