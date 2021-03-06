# This software is distributed under the 3-clause BSD License.
# APH
"""
TBD: dlw june 2020 look at this code in phbase:
            if spcomm is not None: 
                spcomm.sync_with_spokes()
                if spcomm.is_converged():
                    break    

"""


import numpy as np
import math
import re
import shutil
import collections
from pyutilib.misc.timing import TicTocTimer
import time
import logging
import datetime as dt
import mpi4py
import mpi4py.MPI as mpi
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.pysp.phutils import find_active_objective
import mpisppy.utils.listener_util.listener_util as listener_util
import mpisppy.phbase as ph_base  # factor some day...
import mpisppy.utils.sputils as sputils


fullcomm = mpi.COMM_WORLD
rank_global = fullcomm.Get_rank()


logging.basicConfig(level=logging.CRITICAL, # level=logging.CRITICAL, DEBUG
            format='(%(threadName)-10s) %(message)s',
            )


"""
Delete this comment block; dlw May 2019
- deal with "waiting out" a negative tau
z- provide dispatch (z not now)
- add some tests 
"""

""" APH started by DLW, March 2019.
Based on "Algorithm 2: Asynchronous projective hedging (APH) -- Algorithm 1
specialize to the setup S1-S4" from 
"Asynchronous Projective Hedging for Stochastic Programming" 
http://www.optimization-online.org/DB_HTML/2018/10/6895.html
(note: there are therefore some notation changes from PySP1)
Note: we deviate from the paper's notation in the use of i and k
(i is used here as an arbitrary index, usually into the nonants at a node and
k is often used as the "key" (i.e., scenario name) for the local scenarios)
"""

"""
cryptic notes from dlw to dlw:
 ? We don't really want to do all "local_scenarios" every time if there 
is more than one.
cryptic reply from dlw to dlw (April 2019): Ikke n\aa. Sienere.
"""


class APH(ph_base.PHBase):  # ??????
    """
    Args:
        PHoptions (dict): PH options
        all_scenario_names (list): all scenario names
        scenario_creator (fct): returns a concrete model with special things
        scenario_denouement (fct): for post processing and reporting
        all_node_names (list of str): non-leaf node names
        cb_data (any): passed directly to the scenario callback

    Attributes (partial list):
        local_scenarios (dict of scenario objects): concrete models with 
              extra data, key is name
        comms (dict): keys are node names values are comm objects.
        scenario_name_to_rank (dict): all scenario names
        local_scenario_names (list): names of locals 
        current_solver_options (dict): from PHoptions; callbacks might change
        synchronizer (object): asynch listener management
        cb_data (any): passed directly to the scenario callback

    """
    def __init__(self,
                 PHoptions,
                 all_scenario_names,
                 scenario_creator,
                 scenario_denouement,
                 mpicomm=None,
                 all_nodenames=None,
                 cb_data=None,
                 PH_extensions=None, PH_extension_kwargs=None,
                 PH_converger=None, rho_setter=None):
        super().__init__(PHoptions,
                         all_scenario_names,
                         scenario_creator,
                         scenario_denouement,
                         mpicomm=mpicomm,
                         all_nodenames=all_nodenames,
                         cb_data=cb_data,
                         PH_extensions=PH_extensions,
                         PH_extension_kwargs=PH_extension_kwargs,
                         PH_converger=PH_converger,
                         rho_setter=rho_setter)

        self.phis = {} # phi values, indexed by scenario names
        self.nu = 1 # might be changed dynamically by an extension
        self.tau_summand = 0  # place holder for iteration 1 reduce
        self.phi_summand = 0
        self.global_tau = 0
        self.global_phi = 0
        self.global_punorm = 0 # ... may be out of date...
        self.global_pvnorm = 0
        self.global_pwnorm = 0
        self.global_pznorm = 0
        self.local_pwnorm = 0
        self.local_pznorm = 0
        self.conv = None

    #============================
    def setup_Lens(self):
        """ We need to know the lengths of c-style vectors for listener_util
        """
        self.Lens = collections.OrderedDict({"FirstReduce": {},
                                            "SecondReduce": {}})

        for sname, scenario in self.local_scenarios.items():
            for node in scenario._PySPnode_list:
                self.Lens["FirstReduce"][node.name] \
                    = 3 * len(node.nonant_vardata_list)
                self.Lens["SecondReduce"][node.name] = 0 # only use root?
        self.Lens["FirstReduce"]["ROOT"] += self.n_proc  # for time of update
        # tau, phi, punorm, pvnorm, pwnorm, pznorm, secs
        self.Lens["SecondReduce"]["ROOT"] += 6 + self.n_proc 

    #============================
    def Update_y(self, verbose):
        for k,s in self.local_scenarios.items():
            for (ndn,i), xvar in s._nonant_indexes.items():
                # pyo.value vs. _value ??
                xzdiff = xvar._value \
                        - s._zs[(ndn,i)]._value
                s._ys[(ndn,i)]._value = pyo.value(s._Ws[(ndn,i)]) \
                                      + pyo.value(s._PHrho[(ndn,i)]) \
                                      * xzdiff
                if verbose and self.rank == self.rank0:
                    print ("rank, node, scen, var, y", ndn, k,
                           self.rank, xvar.name,
                           pyo.value(s._ys[(ndn,i)]))


    #============================***********=========
    def listener_side_gig(self, synchro):
        """ Called by the listener after the first reduce.
        First, see if there are enough xbar contributions to proceed.
        If there are, then compute tau and phi.
        NOTE: it gets the synchronizer as an arg but self already has it.
        [WIP]
        We are going to disable the side_gig on self if we
        updated tau and phi.
        Massive side-effects: e.g., update xbar etc.
        """
        # This does unsafe things, so it can only be called when the worker is
        # in a tight loop in compute_averages that respects the data lock.

        verbose = self.PHoptions["verbose"]
        # See if we have enough xbars to proceed (need not be perfect)
        xbarin = 0 # count ranks (close enough to be a proxy for scenarios)
        self.synchronizer._unsafe_get_global_data("FirstReduce",
                                                  self.node_concats)
        self.synchronizer._unsafe_get_global_data("SecondReduce",
                                                  self.node_concats)
        # last_phi_tau_update_time
        lptut =  np.max(self.node_concats["SecondReduce"]["ROOT"][6:])
        logging.debug('enter side gig, last phi update={}'.format(lptut))
        for cr in range(self.n_proc):
            backdist = self.n_proc - cr
            logging.debug('*side_gig* cr {} on rank {} time {}'.\
                format(cr, self.rank,
                    self.node_concats["FirstReduce"]["ROOT"][-backdist]))
            if  self.node_concats["FirstReduce"]["ROOT"][-backdist] \
                >= lptut:
                xbarin += 1
        if xbarin/self.n_proc < self.PHoptions["async_frac_needed"]:
            logging.debug('   not enough on rank {}'.format(self.rank))
            # We have not really "done" the side gig.
            return

        # If we are still here, we have enough to do the calculations
        logging.debug('   good to go on rank {}'.format(self.rank))
        if verbose and self.rank == self.rank0:
            print ("(%d)" % xbarin)
            
        # set the xbar, xsqbar, and ybar in all the scenarios
        for k,s in self.local_scenarios.items():
            nlens = s._PySP_nlens        
            for (ndn,i) in s._nonant_indexes:
                s._xbars[(ndn,i)]._value \
                    = self.node_concats["FirstReduce"][ndn][i]
                s._xsqbars[(ndn,i)]._value \
                    = self.node_concats["FirstReduce"][ndn][nlens[ndn]+i]
                s._ybars[(ndn,i)]._value \
                    = self.node_concats["FirstReduce"][ndn][2*nlens[ndn]+i]

                if verbose and self.rank == self.rank0:
                    print ("rank, scen, node, var, xbar:",
                           self.rank,k,ndn,s._nonant_indexes[ndn,i].name,
                           pyo.value(s._xbars[(ndn,i)]))

        # There is one tau_summand for the rank; global_tau is out of date when
        # we get here because we could not compute it until the averages were.
        # vk is just going to be ybar directly
        if not hasattr(self, "uk"):
            self.uk = {} # indexed by sname and nonant index [sname][(ndn,i)]
        self.local_punorm = 0  # local summand for probability weighted norm
        self.local_pvnorm = 0
        new_tau_summand = 0  # for this rank
        for sname,s in self.local_scenarios.items():
            scen_unorm = 0.0
            scen_vnorm = 0.0
            if sname not in self.uk:
                self.uk[sname] = {}
            nlens = s._PySP_nlens        
            for (ndn,i), xvar in s._nonant_indexes.items():
                self.uk[sname][(ndn,i)] = xvar._value \
                                          - pyo.value(s._xbars[(ndn,i)])
                # compute the unorm and vnorm
                scen_unorm += self.uk[sname][(ndn,i)] \
                              * self.uk[sname][(ndn,i)]
                scen_vnorm += pyo.value(s._ybars[(ndn,i)]) \
                              * pyo.value(s._ybars[(ndn,i)])
            self.local_punorm += pyo.value(s.PySP_prob) * scen_unorm
            self.local_pvnorm += pyo.value(s.PySP_prob) * scen_vnorm
            new_tau_summand += pyo.value(s.PySP_prob) \
                               * (scen_unorm + scen_vnorm)
            
        # tauk is the expecation of the sum sum of squares; update for this calc
        logging.debug('  in side-gig, old global_tau={}'.format(self.global_tau))
        logging.debug('  in side-gig, old summand={}'.format(self.tau_summand))
        logging.debug('  in side-gig, new summand={}'.format(new_tau_summand))
        self.global_tau = self.global_tau - self.tau_summand + new_tau_summand
        self.tau_summand = new_tau_summand # make available for the next reduce
        logging.debug('  in side-gig, new global_tau={}'.format(self.global_tau))

        # now we can get the local contribution to the phi_sum 
        if self.global_tau <= 0:
            logging.debug('  *** Negative tau={} on rank {}'\
                          .format(self.global_tau, self.rank))
        self.phi_summand = 0.0
        for k,s in self.local_scenarios.items():
            self.phis[k] = 0.0
            for (ndn,i), xvar in s._nonant_indexes.items():
                ### xvar = node.nonant_vardata_list[i] hideous bug... delete this
                self.phis[k] += (pyo.value(s._zs[(ndn,i)]) - xvar._value) \
                    *(pyo.value(s._Ws[(ndn,i)]) - pyo.value(s._ys[(ndn,i)]))
            self.phis[k] *= pyo.value(s.PySP_prob)
            self.phi_summand += self.phis[k]

        # prepare for the reduction that will take place after this side-gig
        self.local_concats["SecondReduce"]["ROOT"][0] = self.tau_summand
        self.local_concats["SecondReduce"]["ROOT"][1] = self.phi_summand
        self.local_concats["SecondReduce"]["ROOT"][2] = self.local_punorm
        self.local_concats["SecondReduce"]["ROOT"][3] = self.local_pvnorm
        self.local_concats["SecondReduce"]["ROOT"][4] = self.local_pwnorm
        self.local_concats["SecondReduce"]["ROOT"][5] = self.local_pznorm
        # we have updated our summands and the listener will do a reduction
        secs_so_far = (dt.datetime.now() - self.startdt).total_seconds()
        # Put in a time only for this rank, so the "sum" is really a report
        self.local_concats["SecondReduce"]["ROOT"][6+self.rank] = secs_so_far
        # This is run by the listener, so don't tell the worker you have done
        # it until you are sure you have.
        self.synchronizer._unsafe_put_local_data("SecondReduce",
                                                 self.local_concats)
        self.synchronizer.enable_side_gig = False  # we did it
        logging.debug(' exit side_gid on rank {}'.format(self.rank))
        
    #============================
    def Compute_Averages(self, verbose=False):
        """Gather ybar, xbar and x squared bar for each node 
           and also distribute the values back to the scenarios.
           Compute the tau summand from self and distribute back tauk
           (tau_k is a scalar and special with respect to snchronizing).
           Compute the phi summand and reduce it.

        Args:
          verbose (boolean): verbose output

        note: this is a long routine because we need a reduce before
              we can do more calcs that need another reduce and I want
              to keep the reduce calls together.
        NOTE: see compute_xbar for more notes.
        note: DLW: think about multi-stage harder (March 2019); e.g. tau and phi

        """
        if not hasattr(self, "local_concats"):
            nodenames = [] # avoid repeated work
            self.local_concats = {"FirstReduce": {}, # keys are tree node names
                             "SecondReduce": {}}
            self.node_concats = {"FirstReduce": {}, # concat of xbar and xsqbar
                             "SecondReduce": {}} 

            # accumulate & concatenate all local contributions before the reduce

            # create the c-style storage for the concats
            for k,s in self.local_scenarios.items():
                nlens = s._PySP_nlens        
                for node in s._PySPnode_list:
                    if node.name not in nodenames:
                        ndn = node.name
                        nodenames.append(ndn)
                        mylen = self.Lens["FirstReduce"][ndn]
                        self.local_concats["FirstReduce"][ndn]\
                            = np.zeros(mylen, dtype='d')
                        self.node_concats["FirstReduce"][ndn]\
                            = np.zeros(mylen, dtype='d')
            # second reduce is tau and phi
            mylen = self.Lens["SecondReduce"]["ROOT"]
            self.local_concats["SecondReduce"]["ROOT"]\
                = np.zeros(mylen, dtype='d') 
            self.node_concats["SecondReduce"]["ROOT"]\
                = np.zeros(mylen, dtype='d')
        else: # concats are here, just zero them out.
            nodenames = []
            for k,s in self.local_scenarios.items():
                nlens = s._PySP_nlens        
                for node in s._PySPnode_list:
                    if node.name not in nodenames:
                        ndn = node.name
                        nodenames.append(ndn)
                        self.local_concats["FirstReduce"][ndn].fill(0)
                        self.node_concats["FirstReduce"][ndn].fill(0)                    
            self.local_concats["SecondReduce"]["ROOT"].fill(0)
            self.node_concats["SecondReduce"]["ROOT"].fill(0)

        # compute the locals and concat them for the first reduce
        for k,s in self.local_scenarios.items():
            nlens = s._PySP_nlens        
            for node in s._PySPnode_list:
                ndn = node.name
                for i in range(nlens[node.name]):
                    v_value = node.nonant_vardata_list[i]._value
                    self.local_concats["FirstReduce"][node.name][i] += \
                        (s.PySP_prob / node.cond_prob) * v_value
                    self.local_concats["FirstReduce"][node.name][nlens[ndn]+i]\
                        += (s.PySP_prob / node.cond_prob) * v_value * v_value
                    self.local_concats["FirstReduce"][node.name][2*nlens[ndn]+i]\
                        += (s.PySP_prob / node.cond_prob) \
                           * pyo.value(s._ys[(node.name,i)])

        # record the time
        secs_sofar = (dt.datetime.now() - self.startdt).total_seconds()
        # only this rank puts a time for this rank, so the sum is a report
        self.local_concats["FirstReduce"]["ROOT"][3*nlens["ROOT"]+self.rank] \
            = secs_sofar
        logging.debug('Compute_Averages at secs_sofar {} on rank {}'\
                      .format(secs_sofar, self.rank))
                    
        self.synchronizer.compute_global_data(self.local_concats,
                                              self.node_concats,
                                              enable_side_gig = True,
                                              rednames = ["FirstReduce"],
                                              keep_up = True)
        while self.synchronizer.global_quitting == 0 \
              and self.synchronizer.enable_side_gig:
            # Other ranks could be reporting, so keep looking for them.
            self.synchronizer.compute_global_data(self.local_concats,
                                                  self.node_concats)
            if not self.synchronizer.enable_side_gig:
                logging.debug(' did side gig break on rank {}'.format(self.rank))
                break
            else:
                logging.debug('   gig wait sleep on rank {}'.format(self.rank))
                if verbose and self.rank == self.rank0:
                    print ('s'),
                time.sleep(self.PHoptions["async_sleep_secs"])

        # (if the listener still has the lock, compute_global_will wait for it)
        self.synchronizer.compute_global_data(self.local_concats,
                                              self.node_concats)
        # We  assign the global xbar, etc. as side-effect in the side gig, btw
        self.global_tau = self.node_concats["SecondReduce"]["ROOT"][0]
        self.global_phi = self.node_concats["SecondReduce"]["ROOT"][1]
        self.global_punorm = self.node_concats["SecondReduce"]["ROOT"][2]
        self.global_pvnorm = self.node_concats["SecondReduce"]["ROOT"][3]
        self.global_pwnorm = self.node_concats["SecondReduce"]["ROOT"][4]
        self.global_pznorm = self.node_concats["SecondReduce"]["ROOT"][5]

        logging.debug('Assigned global tau {} and phi {} on rank {}'\
                      .format(self.global_tau, self.global_phi, self.rank))
 
    #============================
    def Update_theta_zw(self, verbose):
        """
        Compute and store theta, then update z and w and update
        the probability weighted norms.
        """
        if self.global_tau <= 0:
            logging.debug('|tau {}, rank {}'.format(self.global_tau, self.rank))
            self.theta = 0   
        elif self.global_phi <= 0:
            logging.debug('|phi {}, rank {}'.format(self.global_phi, self.rank))
            self.theta = 0
        else:
            self.theta = self.global_phi * self.nu / self.global_tau
        logging.debug('Assigned theta {} on rank {}'\
                      .format(self.theta, self.rank))

        oldpw = self.local_pwnorm
        oldpz = self.local_pznorm
        self.local_pwnorm = 0
        self.local_pznorm = 0
        # v is just ybar
        for k,s in self.local_scenarios.items():
            probs = pyo.value(s.PySP_prob)
            for (ndn, i) in s._nonant_indexes:
                Ws = pyo.value(s._Ws[(ndn,i)]) + self.theta*self.uk[k][(ndn,i)]
                s._Ws[(ndn,i)] = Ws 
                self.local_pwnorm += probs * Ws * Ws
                zs = pyo.value(s._zs[(ndn,i)])\
                     + self.theta * pyo.value(s._ybars[(ndn,i)])
                s._zs[(ndn,i)] = zs 
                self.local_pznorm += probs * zs * zs
        # ? so they will be there next time? (we really need a third reduction)
        self.local_concats["SecondReduce"]["ROOT"][4] = self.local_pwnorm
        self.local_concats["SecondReduce"]["ROOT"][5] = self.local_pznorm
        # The values we just computed can't be in the global yet, so update here
        self.global_pwnorm += (self.local_pwnorm - oldpw)
        self.global_pznorm += (self.local_pznorm - oldpz)
                
    #============================
    def Compute_Convergence(self, verbose=False):
        """
        The convergence metric is the sqrt of the sum of
        probability weighted unorm scaled by the probability weighted w norm
        probability weighted vnorm scaled by the probability weighted z norm
 
        Returns:
            update self.conv if appropriate
        """
        # dlw to dlw, April 2019: wnorm and znorm are in update_zw;
        # the u and v should be in the side gig.
        # you need a reduction on all the norms!!

        if self.global_pwnorm > 0 and self.global_pznorm > 0:
            self.conv = math.sqrt(self.global_punorm / self.global_pwnorm \
                                  + self.global_pvnorm / self.global_pznorm)
        logging.debug('self.conv={} self.global_punorm={} self.global_pwnorm={} self.global_pvnorm={} self.global_pznorm={})'\
                      .format(self.conv, self.global_punorm, self.global_pwnorm, self.global_pvnorm, self.global_pznorm))
        # allow a PH converger, mainly for mpisspy to get xhat from a wheel conv
        # It probably cannot get a lower bound or even try to
        if hasattr(self, "PH_conobject") and self.PH_convobject is not None:
            phc = self.PH_convobject(self, self.rank, self.n_proc)
            logging.debug("PH converger called (returned {})".format(phc))


    #====================================================================
    def APH_iterk(self, spcomm):
        """ Loop for the main iterations (called by synchronizer).

        Args:
        spcomm (SPCommunitator object): to communicate intra and inter

        Updates: 
            self.conv (): APH convergence

        """
        logging.debug('==== enter iterk on rank {}'.format(self.rank))
        verbose = self.PHoptions["verbose"]
        have_extensions = self.PH_extensions is not None
        have_converger = self.PH_converger is not None
        dprogress = self.PHoptions["display_progress"]
        dtiming = self.PHoptions["display_timing"] 
        self.conv = None
        # we enter after the iteration 0 solves, so do updates first
        for self._PHIter in range(1, int(self.PHoptions["PHIterLimit"])+1):
            iteration_start_time = time.time()

            if dprogress and self.rank == self.rank0:
                print("")
                print ("Initiating APH Iteration",self._PHIter)
                print("")

            self.Update_y(verbose)
            # Compute xbar, etc
            logging.debug('pre Compute_Averages on rank {}'.format(self.rank))
            self.Compute_Averages(verbose)
            logging.debug('post Compute_Averages on rank {}'.format(self.rank))
            if self.global_tau <= 0:
                logging.debug('***tau is 0 on rank {}'.format(self.rank))

            # Apr 2019 dlw: If you want the convergence crit. to be up to date,
            # do this as a listener side-gig and add another reduction.
            self.Update_theta_zw(verbose)
            self.Compute_Convergence()  # updates conv

            # The hub object takes precedence 
            # over the converger
            #### mpi4py.MPI.Intracomm
            if spcomm is not None and type(spcomm) is not mpi4py.MPI.Intracomm:
                spcomm.sync_with_spokes()
                if spcomm.is_converged():
                    break    
            if have_converger:
                if self.convobject.is_converged():
                    converged = True
                    if self.rank == self.rank0:
                        print("User-supplied converger determined termination criterion reached")
                    break
            
            # slight divergence from PH, where mid-iter is before conv
            if have_extensions:
                self.extobject.miditer()
            
            teeme = ("tee-rank0-solves" in self.PHoptions) \
                 and (self.PHoptions["tee-rank0-solves"] == True)
            # Let the solve loop deal with persistent solvers & signal handling
            self.solve_loop(solver_options = \
                            self.current_solver_options,
                            dtiming = dtiming,
                            gripe = True,
                            disable_pyomo_signal_handling = True,
                            tee = teeme,
                            verbose = verbose)

            if have_extensions:
                self.extobject.enditer()

            if dprogress and self.rank == self.rank0:
                print("")
                print("After APH Iteration",self._PHIter)
                print("Convergence Metric=",self.conv)
                print("Iteration time: %6.2f" \
                      % (time.time() - iteration_start_time))
                print("Elapsed time:   %6.2f" \
                      % (dt.datetime.now() - self.startdt).total_seconds())

        logging.debug('Setting synchronizer.quitting on rank %d' % self.rank)
        self.synchronizer.quitting = 1

    #====================================================================
    def APH_main(self, spcomm=None):

        """Execute the APH algorithm.
        Args:
            spcomm (SPCommunitator object): for intra or inter communications

        Returns:
            conv, Eobj, trivial_bound: 
                        The first two CANNOT BE EASILY INTERPRETED. 
                        Eobj is the expected,  weighted objective with 
                        proximal term. It is not directly useful.
                        The trivial bound is computed after iter 0
        NOTE:
            You need an xhat finder either in denoument or in an extension.
        """
        # Prep needs to be before iter 0 for bundling
        # (It could be split up)
        self.PH_Prep(attach_prox=False)

        # Begin APH-specific Prep
        for sname, scenario in self.local_scenarios.items():    
            # ys is plural of y
            scenario._ys = pyo.Param(scenario._nonant_indexes.keys(),
                                        initialize = 0.0,
                                        mutable = True)
            scenario._ybars = pyo.Param(scenario._nonant_indexes.keys(),
                                        initialize = 0.0,
                                        mutable = True)
            scenario._zs = pyo.Param(scenario._nonant_indexes.keys(),
                                        initialize = 0.0,
                                        mutable = True)
                
            objfct = find_active_objective(scenario, True)
                
            for (ndn,i), xvar in scenario._nonant_indexes.items():
                # proximal term
                objfct.expr +=  scenario._PHprox_on[(ndn,i)] * \
                    (scenario._PHrho[(ndn,i)] /2.0) * \
                    (xvar - scenario._zs[(ndn,i)]) * \
                    (xvar - scenario._zs[(ndn,i)])

        # End APH-specific Prep
        
        self.subproblem_creation(self.PHoptions["verbose"])
        
        trivial_bound = self.Iter0()
        self.setup_Lens()

        sleep_secs = self.PHoptions["async_sleep_secs"]

        lkwargs = None  # nothing beyond synchro
        listener_gigs = {"FirstReduce": (self.listener_side_gig, lkwargs),
                         "SecondReduce": None}
        self.synchronizer = listener_util.Synchronizer(comms = self.comms,
                                                    Lens = self.Lens,
                                                    work_fct = self.APH_iterk,
                                                    rank = self.rank,
                                                    sleep_secs = sleep_secs,
                                                    asynch = True,
                                                    listener_gigs = listener_gigs)
        args = [spcomm.intracomm] if spcomm is not None else [fullcomm]
        kwargs = None  # {"PH_extensions": PH_extensions}
        self.synchronizer.run(args, kwargs)

        Eobj = self.post_loops()

        return self.conv, Eobj, trivial_bound

#************************************************************
if __name__ == "__main__":
    # hardwired by dlw for debugging
    import mpisppy.examples.farmer.farmer as refmodel

    PHopt = {}
    PHopt["asynchronousPH"] = False # APH is *projective* and always APH
    PHopt["solvername"] = "cplex"
    PHopt["PHIterLimit"] = 5
    PHopt["defaultPHrho"] = 1
    PHopt["convthresh"] = 0.001
    PHopt["verbose"] = True
    PHopt["display_timing"] = True
    PHopt["display_progress"] = True
    # one way to set up options (never mind that this is not a MIP)
    PHopt["iter0_solver_options"] = sputils.option_string_to_dict("mipgap=0.01")
    # another way
    PHopt["iterk_solver_options"] = {"mipgap": 0.001}

    ScenCount = 3
    cb_data={'use_integer': False, "CropsMult": 1}
    all_scenario_names = list()
    for sn in range(ScenCount):
        all_scenario_names.append("scen"+str(sn))
    # end hardwire

    scenario_creator = refmodel.scenario_creator
    scenario_denouement = refmodel.scenario_denouement

    PHopt["async_frac_needed"] = 0.5
    PHopt["async_sleep_secs"] = 0.5
    aph = APH(PHopt, all_scenario_names, scenario_creator, scenario_denouement,
              cb_data=cb_data)


    """
    import xhatlooper
    PHopt["xhat_looper_options"] =  {"xhat_solver_options":\
                                     PHopt["iterk_solver_options"],
                                     "scen_limit": 3,
                                     "csvname": "looper.csv"}
    """
    conv, obj, bnd = aph.APH_main()

    if aph.rank == aph.rank0:
        print ("E[obj] for converged solution (probably NOT non-anticipative)",
               obj)

    dopts = sputils.option_string_to_dict("mipgap=0.001")
    objbound = aph.post_solve_bound(solver_options=dopts, verbose=False)
    if (aph.rank == aph.rank0):
        print ("**** Lagrangian objective function bound=",objbound)
        print ("(probably converged way too early, BTW)")
