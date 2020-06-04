# updated April 2020
import mpisppy.cylinders.spoke as spoke
from mpisppy.extensions.xhatlooper import XhatLooper
import logging
import mpisppy.log

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.xhatlooper_bounder",
                         "xhatlp.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.cylinders.xhatlooper_bounder")


class XhatLooperInnerBound(spoke.InnerBoundNonantSpoke):

    def xhatlooper_prep(self):
        verbose = self.opt.options['verbose']
        if "bundles_per_rank" in self.opt.options\
           and self.opt.options["bundles_per_rank"] != 0:
            raise RuntimeError("xhat spokes cannot have bundles (yet)")
            
        xhatter = XhatLooper(self.opt)

        self.opt.PH_Prep()  
        logger.debug(f"  xhatlooper spoke back from PH_Prep rank {self.rank_global}")

        self.opt.subproblem_creation(verbose)

        ### begin iter0 stuff
        xhatter.pre_iter0()
        self.opt._save_original_nonants()
        self.opt._create_solvers()

        teeme = False
        if "tee-rank0-solves" in self.opt.options:
            teeme = self.opt.options['tee-rank0-solves']

        self.opt.solve_loop(
            solver_options=self.opt.current_solver_options,
            dtiming=False,
            gripe=True,
            tee=teeme,
            verbose=verbose
        )
        self.opt._update_E1()  # Apologies for doing this after the solves...
        if abs(1 - self.opt.E1) > self.opt.E1_tolerance:
            if self.rank_global == self.opt.rank0:
                print("ERROR")
                print("Total probability of scenarios was ", self.opt.E1)
                print("E1_tolerance = ", self.opt.E1_tolerance)
            quit()
        feasP = self.opt.feas_prob()
        if feasP != self.opt.E1:
            if self.rank_global == self.opt.rank0:
                print("ERROR")
                print("Infeasibility detected; E_feas, E1=", feasP, self.opt.E1)
            quit()
        ### end iter0 stuff

        xhatter.post_iter0()
        self.opt._save_nonants() # make the cache

        return xhatter

    def _populate_nonant_caches(self):
        ''' We could use split, but I will use a loop to split scenarios
            This is a hack to use the _PySP_nonant_cache

            DTM: Does this function exist in PHBase?
        '''
        opt = self.opt
        ci = 0 # index into source
        for s in opt.local_scenarios.values():
            itarget = 0 # index into target
            for node in s._PySPnode_list:
                for i in range(s._PySP_nlens[node.name]):
                    try:
                        s._PySP_nonant_cache[itarget] = self.localnonants[ci]
                    except IndexError as e:
                        print("itarget={}, ci={}".format(itarget, ci))
                        raise e
                    itarget += 1
                    ci += 1

    def main(self):
        verbose = self.opt.options["verbose"] # typing aid  
        logger.debug(f"Entering main on xhatlooper spoke rank {self.rank_global}")

        xhatter = self.xhatlooper_prep()

        scen_limit = self.opt.options['xhat_looper_options']['scen_limit']

        xh_iter = 1
        while not self.got_kill_signal():
            if (xh_iter-1) % 10000 == 0:
                logger.debug(f'   Xhatlooper loop iter={xh_iter} on rank {self.rank_global}')
                logger.debug(f'   Xhatlooper got from opt on rank {self.rank_global}')

            if self.new_nonants:
                logger.debug(f'   *Xhatlooper loop iter={xh_iter}')
                logger.debug(f'   *got a new one! on rank {self.rank_global}')
                logger.debug(f'   *localnonants={str(self.localnonants)}')

                self._populate_nonant_caches()
                self.opt._restore_nonants()
                upperbound, srcsname = xhatter.xhat_looper(scen_limit=scen_limit)

                # send a bound to the opt companion
                if upperbound is not None:
                    self.bound = upperbound
                    logger.debug(f'   send inner bound={upperbound} on rank {self.rank_global} (based on scenario {srcsname})')
                logger.debug(f'   bottom of xhatlooper loop on rank {self.rank_global}')
            xh_iter += 1