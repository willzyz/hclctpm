import numpy as np 
import matplotlib.pyplot as plt 
import pickle as pkl 

class Experimentation: 
    
    ## general experimentation class 
    ## for promotion targeting model 
    def compute_aucc(self, ics, ios): 
        assert(len(ics) == len(ios)) 
        curve_area = 0 
        for i in range(len(ios) - 1): 
            cur_area = 0.5 * (ios[i] + ios[i + 1]) * (ics[i + 1] - ics[i]) 
            curve_area = curve_area + cur_area 
        rectangle_area = ics[-1] * ios[-1] 
        aucc = 1.0 * curve_area / rectangle_area 
        if np.isnan(aucc): 
            import ipdb; ipdb.set_trace() 
        
        return aucc 
    
    def AUC_cpit_cost_curve_deciles_cohort_vis(self, pred_values, values, w, n9d_ni_usd, color, plot_random=False): 
        ## function plots the cost curve the slope of the curve correspond to 
        ## inverse of CPIT as a growth metric 
        ## [TODO] plan to compute Area Under Curve (AUC) 
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   n9d_ni_usd: list of next-9-day net inflow in usd per user         
        
        ValuesControl = values[w < 0.5] 
        NIControl = n9d_ni_usd[w < 0.5] 
        lenControl = len(ValuesControl) 
        
        rpu_control = np.sum(1.0 * ValuesControl) / lenControl 
        
        nipu_control = np.sum(1.0 * NIControl) / lenControl 
        
        print('rpu_control: ' + str(rpu_control)) 
        print('nipu_control: ' + str(nipu_control))
        
        ValuesFT = values[w > 0.5] 
        NIFT = n9d_ni_usd[w > 0.5] 
        lenFT = len(ValuesFT) 
        
        rpu_ft = np.sum(1.0 * ValuesFT) / lenFT

        nipu_ft = np.sum(1.0 * NIFT) / lenFT 

        print('rpu_ft: ' + str(rpu_ft)) 
        print('nipu_ft: ' + str(nipu_ft))
        
        ios = [] 
        ics = [] 
        iopus = [] 
        icpus = [] 
        percs = [] ## let's keep percentages of population, for cross checks 
        cpits = []
        cpitcohorts = []
        treatment_ratios = [] 
        
        ## rewriting the code so that we have an inverse rank 
        pred_values = np.reshape(pred_values, (-1, 1)) 
        indices = np.reshape(range(pred_values.shape[0]), (-1, 1)) 
        
        TD = np.concatenate((pred_values, indices), axis = 1) 
        TD = TD[(-1.0 * TD[:, 0]).argsort(), :] 
        #print('printing out the model scores') 
        #print(TD[0:5])
        ## produce the threshold using a percentage (14%) of users to target 
        #threshold_index = int(0.14 * len(TD)) 
        #threshold = TD[threshold_index, 0] 
        
        """ 
        print('the threshold for choosing 14% of users is: ') 
        print(threshold) 
        print('max: ') 
        print(max(TD[:, 0])) 
        print('min: ') 
        print(min(TD[:, 0])) 
        print('average: ') 
        print(np.mean(TD[:, 0]))
        #exit() 
        """ 
        
        num_segments = 10 
        cnt = 0 
        for idx in range(0, TD.shape[0], int(np.floor(TD.shape[0] / num_segments))): 
            if idx == 0: 
                io = 0 
                ic = 0 
                iopu = 0 
                icpu = 0 
                num_users = 0 
                treatment_ratio = 0 
            else: 
                selected_user_indices = TD[0:idx, 1] 
                
                model_target_users = np.zeros(w.shape) 
                model_target_users[np.reshape(selected_user_indices, (idx,)).astype(np.int)] = 1 ## index 
                
                treated_targeted_filter = np.logical_and(model_target_users, w) 
                untreated_targeted_filter = np.logical_and(model_target_users, (w < 0.5)) 
                treated_untargeted_filter = np.logical_and(model_target_users < 0.5, w ) 
                untreated_untargeted_filter = np.logical_and(model_target_users < 0.5, (w < 0.5)) 
                
                perc = sum(1.0 * model_target_users) / len(model_target_users) 
                
                treated_target_rpu = sum(1.0 * values[treated_targeted_filter]) / sum(treated_targeted_filter) 
                treated_target_nipu = sum(1.0 * n9d_ni_usd[treated_targeted_filter]) / sum(treated_targeted_filter) 
                
                untreated_target_rpu = sum(1.0 * values[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                untreated_target_nipu = sum(1.0 * n9d_ni_usd[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                
                treated_untarget_rpu = sum(1.0 * values[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                treated_untarget_nipu = sum(1.0 * n9d_ni_usd[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                
                untreated_untarget_rpu = sum(1.0 * values[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                untreated_untarget_nipu = sum(1.0 * n9d_ni_usd[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                
                if perc > 0.99: 
                    print('rpu_cohort') 
                    print(rpu_cohort) 
                    print('treated_target_rpu') 
                    print(treated_target_rpu) 
                    rpu_cohort = treated_target_rpu 
                    nipu_cohort = treated_target_nipu 
                else: 
                    rpu_cohort = treated_target_rpu * perc + untreated_untarget_rpu * (1 - perc) 
                    nipu_cohort = treated_target_nipu * perc + untreated_untarget_nipu * (1 - perc) 
                
                iopu = max(rpu_cohort - rpu_control, 1e-7) 
                icpu = max(-1.0 * (nipu_cohort - nipu_control), 1e-7) 
                
                io = iopu * lenControl 
                ic = icpu * lenControl 
                
                cpit = icpu / iopu 
                
                #cpitpc = -1.0 * (treated_target_nipu - nipu_control) / (treated_target_rpu - rpu_control) 
                cpitcohort = -1.0 * (nipu_cohort - nipu_control) / (rpu_cohort - rpu_control) 

                percs.append(perc)
                cpits.append(cpit)
                cpitcohorts.append(cpitcohort)
                
                liftvscontrol = (rpu_cohort - rpu_control) * 1.0 / (rpu_control) 
                liftrandomvscontrol = (rpu_ft - rpu_control) * 1.0 / (rpu_control) 
                liftpcvscontrol = (treated_target_rpu - rpu_control) * 1.0 / (rpu_control) 
                
                if True: #perc > 0.35 and perc < 0.45: 
                    print('---------------------------->>>>>>') 
                    print('perc - target: %.2f' % perc) 
                    print('treated_target_rpu: %.2f' % treated_target_rpu) 
                    print('treated_target_nipu: %.2f' %treated_target_nipu) 
                    print('nontreated_target_rpu: %.2f' % untreated_target_rpu) 
                    print('nontreated_target_nipu: %.2f' % untreated_target_nipu) 
                    print('treated_nontarget_rpu: %.2f' %treated_untarget_rpu) 
                    print('treated_nontarget_nipu: %.2f' %treated_untarget_nipu) 
                    print('nontreated_nontarget_rpu: %.2f' %untreated_untarget_rpu) 
                    print('nontreated_nontarget_nipu: %.2f' %untreated_untarget_nipu) 
                    
                    cpit_targeted = -1.0 * (treated_target_nipu - untreated_target_nipu) / (treated_target_rpu - untreated_target_rpu) 
                    cpit_nontargeted = -1.0 * (treated_untarget_nipu - untreated_untarget_nipu) / (treated_untarget_rpu - untreated_untarget_rpu) 
                    
                    print('--- with ' + str(perc * 100) + '% targeting, print cpits to treat users and create incrementality in users ---') 
                    print('--> in targeted users: ') 
                    print('cpit = ' + str(cpit_targeted)) 
                    print('--> in non-targeted users: ') 
                    print('cpit = ' + str(cpit_nontargeted)) 
                    
                    print('rpu_control: %.2f' %rpu_control) 
                    print('nipu_control: %.2f' %nipu_control) 
                    print('rpu_ft: %.2f' %rpu_ft) 
                    print('nipu_ft: %.2f' %nipu_ft) 
                    print('rpu_cohort: %.2f' %rpu_cohort) 
                    print('nipu_cohort: %.2f' %nipu_cohort) 
                    
                    print('lift targeted cohort vs control: %.2f' %liftvscontrol) 
                    print('lift random vs control: %.2f' %liftrandomvscontrol) 
                    print('cpit cohort vs control: %.2f' %cpit) 
                    print('lift targeted-treated vs control: %.2f' %liftpcvscontrol) 
                    print('cpit cohort: %2f' %cpitcohort) 
            
            ics.append(ic) 
            ### ---- guard against the cost curve going down --- 
            ### for plotting and visualization 
            if len(ios) > 0: 
                if io < ios[-1]: 
                    io = ios[-1] 
            ios.append(io) 
            
            icpus.append(icpu) 
            iopus.append(iopu) 
            treatment_ratios.append(treatment_ratio) 
            
            """ 
            print('decile: ' + str(cnt)) 
            print('cost: ' + str(ic)) 
            print('increment orders: ' + str(io)) 
            print('increment orders per user: ' + str(iopu)) 
            if io != 0: 
                print('cpit: ' + str(ic * 1.0 / io)) 
            """ 
            cnt = cnt + 1 
        
        ## sort the tuple by its index (cost) 
        ics = np.asarray(ics); ics = np.reshape(ics, (-1, 1)) 
        ios = np.asarray(ios); ios = np.reshape(ios, (-1, 1)) 
        ips = np.minimum(ics / ics[-1], 1.0) 
        ios = np.minimum(ios / ios[-1], 1.0) 
        
        if plot_random == True: 
            ips = np.reshape(np.arange(0.0, 1.1, 0.1), (-1, 1)) 
            ios = np.reshape(np.arange(0.0, 1.1, 0.1), (-1, 1)) 
        
        ### to guarantee to remove nan issues 
        ips[-1] = 1.0 
        ios[-1] = 1.0 
        
        #ips = ics 
        #ios = ios 
        combined_series = np.concatenate((ics, ios), axis=1) 
        combined_series = np.concatenate((combined_series, ips), axis=1) 
        
        #combined_series = combined_series[combined_series[:, 0].argsort(), :] 
        ## plotting on different colors on same figure 
        ## -- combined_series [:, 0] contains cost 
        ## -- combined_series [:, 1] contains trips/orders 
        ## -- combined_series [:, 2] contains percentages 
        
        ### [for Hong:] feel free to save this to file 
        ### let's define the interface between eng and vis 
        # plt.plot(combined_series[:, 2], combined_series[:, 1], '-o'+color, markersize=12, linewidth=3)
        ### [Todo:] define the file format
        ### ask Bhavya about this evaluator interface
        
        aucc = self.compute_aucc(ips, ios) 
        # plt.xlabel('Incremental cost % of maximum')
        # plt.ylabel('Incremental value % of maximum')

        # ax = plt.gca()
        # type(ax)  # matplotlib.axes._subplots.AxesSubplot

        # # manipulate to use percentage
        # vals = ax.get_xticks()
        # #ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
        # vals = ax.get_yticks()
        # #ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
        # plt.grid(True)
        return aucc, percs, cpits, cpitcohorts
    
    def AUC_cpit_cost_curve_deciles_cohort(self, pred_values, values, w, n9d_ni_usd, color, plot_random=False): 
        ## function plots the cost curve the slope of the curve correspond to 
        ## inverse of CPIT as a growth metric 
        ## [TODO] plan to compute Area Under Curve (AUC) 
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   n9d_ni_usd: list of next-9-day net inflow in usd per user         
        
        ValuesControl = values[w < 0.5] 
        NIControl = n9d_ni_usd[w < 0.5] 
        lenControl = len(ValuesControl) 
        
        rpu_control = np.sum(1.0 * ValuesControl) / lenControl 
        
        nipu_control = np.sum(1.0 * NIControl) / lenControl 
        
        print('rpu_control: ' + str(rpu_control)) 
        print('nipu_control: ' + str(nipu_control))
        
        ValuesFT = values[w > 0.5] 
        NIFT = n9d_ni_usd[w > 0.5] 
        lenFT = len(ValuesFT) 
        
        rpu_ft = np.sum(1.0 * ValuesFT) / lenFT
        
        nipu_ft = np.sum(1.0 * NIFT) / lenFT 
        
        print('rpu_ft: ' + str(rpu_ft)) 
        print('nipu_ft: ' + str(nipu_ft))
        
        ios = [] 
        ics = [] 
        iopus = [] 
        icpus = [] 
        percs = [] ## let's keep percentages of population, for cross checks 
        cpits = []
        cpitcohorts = []
        treatment_ratios = [] 
        
        ## rewriting the code so that we have an inverse rank 
        pred_values = np.reshape(pred_values, (-1, 1)) 
        indices = np.reshape(range(pred_values.shape[0]), (-1, 1)) 
        
        TD = np.concatenate((pred_values, indices), axis = 1) 
        TD = TD[(-1.0 * TD[:, 0]).argsort(), :] 
        #print('printing out the model scores') 
        #print(TD[0:5])
        ## produce the threshold using a percentage (14%) of users to target 
        #threshold_index = int(0.14 * len(TD)) 
        #threshold = TD[threshold_index, 0] 
        
        """ 
        print('the threshold for choosing 14% of users is: ') 
        print(threshold) 
        print('max: ') 
        print(max(TD[:, 0])) 
        print('min: ') 
        print(min(TD[:, 0])) 
        print('average: ') 
        print(np.mean(TD[:, 0]))
        #exit() 
        """ 
        
        num_segments = 20 
        cnt = 0 
        for idx in range(0, TD.shape[0], int(np.floor(TD.shape[0] / num_segments))): 
            if idx == 0: 
                io = 0 
                ic = 0 
                iopu = 0 
                icpu = 0 
                num_users = 0 
                treatment_ratio = 0 
            else: 
                selected_user_indices = TD[0:idx, 1] 
                
                model_target_users = np.zeros(w.shape) 
                model_target_users[np.reshape(selected_user_indices, (idx,)).astype(np.int)] = 1 ## index 
                
                treated_targeted_filter = np.logical_and(model_target_users, w) 
                untreated_targeted_filter = np.logical_and(model_target_users, (w < 0.5)) 
                treated_untargeted_filter = np.logical_and(model_target_users < 0.5, w ) 
                untreated_untargeted_filter = np.logical_and(model_target_users < 0.5, (w < 0.5)) 
                
                perc = sum(1.0 * model_target_users) / len(model_target_users) 
                
                treated_target_rpu = sum(1.0 * values[treated_targeted_filter]) / sum(treated_targeted_filter) 
                treated_target_nipu = sum(1.0 * n9d_ni_usd[treated_targeted_filter]) / sum(treated_targeted_filter) 
                
                untreated_target_rpu = sum(1.0 * values[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                untreated_target_nipu = sum(1.0 * n9d_ni_usd[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                
                treated_untarget_rpu = sum(1.0 * values[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                treated_untarget_nipu = sum(1.0 * n9d_ni_usd[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                
                untreated_untarget_rpu = sum(1.0 * values[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                untreated_untarget_nipu = sum(1.0 * n9d_ni_usd[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                
                if perc > 0.99: 
                    print('rpu_cohort') 
                    print(rpu_cohort) 
                    print('treated_target_rpu') 
                    print(treated_target_rpu) 
                    rpu_cohort = treated_target_rpu 
                    nipu_cohort = treated_target_nipu 
                else: 
                    rpu_cohort = treated_target_rpu * perc + untreated_untarget_rpu * (1 - perc) 
                    nipu_cohort = treated_target_nipu * perc + untreated_untarget_nipu * (1 - perc) 
                
                iopu = max(rpu_cohort - rpu_control, 1e-7) 
                icpu = max(-1.0 * (nipu_cohort - nipu_control), 1e-7) 
                
                io = iopu * lenControl 
                ic = icpu * lenControl 
                
                cpit = icpu / iopu 
                
                #cpitpc = -1.0 * (treated_target_nipu - nipu_control) / (treated_target_rpu - rpu_control) 
                cpitcohort = -1.0 * (nipu_cohort - nipu_control) / (rpu_cohort - rpu_control) 

                percs.append(perc)
                cpits.append(cpit)
                cpitcohorts.append(cpitcohort)
                
                liftvscontrol = (rpu_cohort - rpu_control) * 1.0 / (rpu_control) 
                liftrandomvscontrol = (rpu_ft - rpu_control) * 1.0 / (rpu_control) 
                liftpcvscontrol = (treated_target_rpu - rpu_control) * 1.0 / (rpu_control) 
                
                if True: #perc > 0.35 and perc < 0.45: 
                    print('---------------------------->>>>>>') 
                    print('perc - target: %.2f' % perc) 
                    print('treated_target_rpu: %.2f' % treated_target_rpu) 
                    print('treated_target_nipu: %.2f' %treated_target_nipu) 
                    print('nontreated_target_rpu: %.2f' % untreated_target_rpu) 
                    print('nontreated_target_nipu: %.2f' % untreated_target_nipu) 
                    print('treated_nontarget_rpu: %.2f' %treated_untarget_rpu) 
                    print('treated_nontarget_nipu: %.2f' %treated_untarget_nipu) 
                    print('nontreated_nontarget_rpu: %.2f' %untreated_untarget_rpu) 
                    print('nontreated_nontarget_nipu: %.2f' %untreated_untarget_nipu) 
                    
                    cpit_targeted = -1.0 * (treated_target_nipu - untreated_target_nipu) / (treated_target_rpu - untreated_target_rpu) 
                    cpit_nontargeted = -1.0 * (treated_untarget_nipu - untreated_untarget_nipu) / (treated_untarget_rpu - untreated_untarget_rpu) 
                    
                    print('--- with ' + str(perc * 100) + '% targeting, print cpits to treat users and create incrementality in users ---') 
                    print('--> in targeted users: ') 
                    print('cpit = ' + str(cpit_targeted)) 
                    print('--> in non-targeted users: ') 
                    print('cpit = ' + str(cpit_nontargeted)) 
                    
                    print('rpu_control: %.2f' %rpu_control) 
                    print('nipu_control: %.2f' %nipu_control) 
                    print('rpu_ft: %.2f' %rpu_ft) 
                    print('nipu_ft: %.2f' %nipu_ft) 
                    print('rpu_cohort: %.2f' %rpu_cohort) 
                    print('nipu_cohort: %.2f' %nipu_cohort) 
                    
                    print('lift targeted cohort vs control: %.2f' %liftvscontrol) 
                    print('lift random vs control: %.2f' %liftrandomvscontrol) 
                    print('cpit cohort vs control: %.2f' %cpit) 
                    print('lift targeted-treated vs control: %.2f' %liftpcvscontrol) 
                    print('cpit cohort: %2f' %cpitcohort) 
            
            ics.append(ic) 
            ### ---- guard against the cost curve going down --- 
            ### for plotting and visualization 
            if len(ios) > 0: 
                if io < ios[-1]: 
                    io = ios[-1] 
            ios.append(io) 
            
            icpus.append(icpu) 
            iopus.append(iopu) 
            treatment_ratios.append(treatment_ratio) 
            
            """ 
            print('decile: ' + str(cnt)) 
            print('cost: ' + str(ic)) 
            print('increment orders: ' + str(io)) 
            print('increment orders per user: ' + str(iopu)) 
            if io != 0: 
                print('cpit: ' + str(ic * 1.0 / io)) 
            """ 
            cnt = cnt + 1 
        
        ## sort the tuple by its index (cost) 
        ics = np.asarray(ics); ics = np.reshape(ics, (-1, 1)) 
        ios = np.asarray(ios); ios = np.reshape(ios, (-1, 1)) 
        ips = np.minimum(ics / ics[-1], 1.0) 
        ios = np.minimum(ios / ios[-1], 1.0) 
        
        if plot_random == True: 
            ips = np.reshape(np.arange(0.0, 1.0 + 1.0 / num_segments, 1.0 / num_segments), (-1, 1)) 
            ios = np.reshape(np.arange(0.0, 1.0 + 1.0 / num_segments, 1.0 / num_segments), (-1, 1)) 
        
        ### to guarantee to remove nan issues 
        ips[-1] = 1.0 
        ios[-1] = 1.0 
        
        #ips = ics 
        #ios = ios 
        combined_series = np.concatenate((ics, ios), axis=1) 
        combined_series = np.concatenate((combined_series, ips), axis=1) 
        
        #combined_series = combined_series[combined_series[:, 0].argsort(), :] 
        ## plotting on different colors on same figure 
        ## -- combined_series [:, 0] contains cost 
        ## -- combined_series [:, 1] contains trips/orders 
        ## -- combined_series [:, 2] contains percentages 
        
        ### [for Hong:] feel free to save this to file 
        ### let's define the interface between eng and vis 
        plt.plot(combined_series[:, 2], combined_series[:, 1], '-o'+color, markersize=12, linewidth=3) 
        
        ### [Todo:] define the file format 
        ### ask Bhavya about this evaluator interface 
        
        aucc = self.compute_aucc(ips, ios) 
        plt.xlabel('Incremental cost % of maximum')
        plt.ylabel('Incremental value % of maximum')
        
        ax = plt.gca()
        # type(ax)  # matplotlib.axes._subplots.AxesSubplot
        
        # # manipulate to use percentage
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
        plt.grid(True)
        return aucc, combined_series[:, 2], combined_series[:, 1] 
    
    def AUC_cpit_cost_curve_deciles_cohort_d2dist(self, pred_values, values, w, n9d_ni_usd, d2dist, d2dlamb, color, plot_random=False): 
        ## function plots the cost curve the slope of the curve correspond to 
        ## inverse of CPIT as a growth metric 
        ## [TODO] plan to compute Area Under Curve (AUC) 
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   n9d_ni_usd: list of next-9-day net inflow in usd per user         
        
        ValuesControl = values[w < 0.5] 
        NIControl = n9d_ni_usd[w < 0.5] 
        D2DControl = d2dist[w < 0.5] 
        lenControl = len(ValuesControl) 
        
        rpu_control = np.sum(1.0 * ValuesControl) / lenControl 
        
        nipu_control = np.sum(1.0 * NIControl) / lenControl 
        dipu_control = np.sum(1.0 * D2DControl) / lenControl 
        
        print('rpu_control: ' + str(rpu_control)) 
        print('nipu_control: ' + str(nipu_control))
        
        ValuesFT = values[w > 0.5] 
        NIFT = n9d_ni_usd[w > 0.5] 
        lenFT = len(ValuesFT) 
        
        rpu_ft = np.sum(1.0 * ValuesFT) / lenFT

        nipu_ft = np.sum(1.0 * NIFT) / lenFT 

        print('rpu_ft: ' + str(rpu_ft)) 
        print('nipu_ft: ' + str(nipu_ft))
        
        ios = [] 
        ics = [] 
        iopus = [] 
        icpus = [] 
        percs = [] ## let's keep percentages of population, for cross checks 
        cpits = []
        cpitcohorts = []
        cmetriccohorts = [] 
        treatment_ratios = [] 
        
        ## rewriting the code so that we have an inverse rank 
        pred_values = np.reshape(pred_values, (-1, 1)) 
        indices = np.reshape(range(pred_values.shape[0]), (-1, 1)) 
        
        TD = np.concatenate((pred_values, indices), axis = 1) 
        TD = TD[(-1.0 * TD[:, 0]).argsort(), :] 
        #print('printing out the model scores') 
        #print(TD[0:5])
        ## produce the threshold using a percentage (14%) of users to target 
        #threshold_index = int(0.14 * len(TD)) 
        #threshold = TD[threshold_index, 0] 
        
        """ 
        print('the threshold for choosing 14% of users is: ') 
        print(threshold) 
        print('max: ') 
        print(max(TD[:, 0])) 
        print('min: ') 
        print(min(TD[:, 0])) 
        print('average: ') 
        print(np.mean(TD[:, 0]))
        #exit() 
        """ 
        
        num_segments = 20 
        cnt = 0 
        idx_pre = 0 
        for idx in range(0, TD.shape[0], int(np.floor(TD.shape[0] / num_segments))): 
            if idx == 0: 
                io = 0 
                ic = 0 
                iopu = 0 
                icpu = 0 
                num_users = 0 
                treatment_ratio = 0 
            else: 
                selected_user_indices = TD[0:idx, 1] #idx_pre:idx, 1] 
                
                model_target_users = np.zeros(w.shape) 
                model_target_users[np.reshape(selected_user_indices, (idx,)).astype(np.int)] = 1 ## index 
                #model_target_users[np.reshape(selected_user_indices, (idx - idx_pre,)).astype(np.int)] = 1 ## index 
                
                treated_targeted_filter = np.logical_and(model_target_users, w) 
                untreated_targeted_filter = np.logical_and(model_target_users, (w < 0.5)) 
                treated_untargeted_filter = np.logical_and(model_target_users < 0.5, w ) 
                untreated_untargeted_filter = np.logical_and(model_target_users < 0.5, (w < 0.5)) 
                
                perc = 1.0 * idx / len(model_target_users) #sum(1.0 * model_target_users)
                
                treated_target_rpu = sum(1.0 * values[treated_targeted_filter]) / sum(treated_targeted_filter) 
                treated_target_nipu = sum(1.0 * n9d_ni_usd[treated_targeted_filter]) / sum(treated_targeted_filter) 
                treated_target_dipu = sum(1.0 * d2dist[treated_targeted_filter]) / sum(treated_targeted_filter) 
                
                untreated_target_rpu = sum(1.0 * values[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                untreated_target_nipu = sum(1.0 * n9d_ni_usd[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                untreated_target_dipu = sum(1.0 * d2dist[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                
                treated_untarget_rpu = sum(1.0 * values[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                treated_untarget_nipu = sum(1.0 * n9d_ni_usd[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                treated_untarget_dipu = sum(1.0 * d2dist[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                
                untreated_untarget_rpu = sum(1.0 * values[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                untreated_untarget_nipu = sum(1.0 * n9d_ni_usd[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                untreated_untarget_dipu = sum(1.0 * d2dist[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                
                idx_pre = idx 
                if perc > 0.99: 
                    print('rpu_cohort') 
                    print(rpu_cohort) 
                    print('treated_target_rpu') 
                    print(treated_target_rpu) 
                    rpu_cohort = treated_target_rpu 
                    nipu_cohort = treated_target_nipu 
                    dipu_cohort = treated_target_dipu 
                else: 
                    rpu_cohort = treated_target_rpu * perc + untreated_untarget_rpu * (1 - perc) 
                    nipu_cohort = treated_target_nipu * perc + untreated_untarget_nipu * (1 - perc) 
                    dipu_cohort = treated_target_dipu * perc + untreated_untarget_dipu * (1 - perc) 
                
                iopu = max(rpu_cohort - rpu_control, 1e-7) 
                icpu = max(-1.0 * (nipu_cohort - nipu_control), 1e-7) 
                idpu = max(dipu_cohort - dipu_control, 1e-7) 
                
                io = iopu * lenControl 
                ic = icpu * lenControl 
                id = idpu * lenControl 
                
                cpit = icpu / iopu 
                
                #cpitpc = -1.0 * (treated_target_nipu - nipu_control) / (treated_target_rpu - rpu_control) 
                #cpitcohort = -1.0 * (nipu_cohort - nipu_control) / (rpu_cohort - rpu_control) 
                #cmetriccohort = -1.0 * (nipu_cohort - nipu_control) / (rpu_cohort - rpu_control) + d2dlamb * (dipu_cohort - dipu_control) 
                #cmetriccohort = (dipu_cohort - dipu_control) * ((rpu_cohort - rpu_control) + d2dlamb * -1.0 * (nipu_cohort - nipu_control)) 
                
                cmetriccohort = (dipu_cohort - dipu_control) * ((rpu_cohort - rpu_control) - d2dlamb * -1.0 * (nipu_cohort - nipu_control)) # incremental cost 
                #(rpu_cohort - rpu_control) * (d2dlamb * (nipu_cohort - nipu_control) + (dipu_cohort - dipu_control)) 
                
                percs.append(perc) 
                cpits.append(cpit) 
                #cpitcohorts.append(cpitcohort) 
                cmetriccohorts.append( cmetriccohort) # 1.0 / (1e-7 + cmetriccohort)) # 
                #cmetriccohorts.append( 1.0 / (1e-7 + cmetriccohort)) # 
                
                liftvscontrol = (rpu_cohort - rpu_control) * 1.0 / (rpu_control) 
                liftrandomvscontrol = (rpu_ft - rpu_control) * 1.0 / (rpu_control) 
                liftpcvscontrol = (treated_target_rpu - rpu_control) * 1.0 / (rpu_control) 
                
                if True: #perc > 0.35 and perc < 0.45: 
                    print('---------------------------->>>>>>') 
                    print('perc - target: %.2f' % perc) 
                    print('treated_target_rpu: %.2f' % treated_target_rpu) 
                    print('treated_target_nipu: %.2f' %treated_target_nipu) 
                    print('nontreated_target_rpu: %.2f' % untreated_target_rpu) 
                    print('nontreated_target_nipu: %.2f' % untreated_target_nipu) 
                    print('treated_nontarget_rpu: %.2f' %treated_untarget_rpu) 
                    print('treated_nontarget_nipu: %.2f' %treated_untarget_nipu) 
                    print('nontreated_nontarget_rpu: %.2f' %untreated_untarget_rpu) 
                    print('nontreated_nontarget_nipu: %.2f' %untreated_untarget_nipu) 
                    
                    cpit_targeted = -1.0 * (treated_target_nipu - untreated_target_nipu) / (treated_target_rpu - untreated_target_rpu) 
                    cpit_nontargeted = -1.0 * (treated_untarget_nipu - untreated_untarget_nipu) / (treated_untarget_rpu - untreated_untarget_rpu) 
                    
                    print('--- with ' + str(perc * 100) + '% targeting, print cpits to treat users and create incrementality in users ---') 
                    print('--> in targeted users: ') 
                    print('cpit = ' + str(cpit_targeted)) 
                    print('--> in non-targeted users: ') 
                    print('cpit = ' + str(cpit_nontargeted)) 
                    
                    print('rpu_control: %.2f' %rpu_control) 
                    print('nipu_control: %.2f' %nipu_control) 
                    print('rpu_ft: %.2f' %rpu_ft) 
                    print('nipu_ft: %.2f' %nipu_ft) 
                    print('rpu_cohort: %.2f' %rpu_cohort) 
                    print('nipu_cohort: %.2f' %nipu_cohort) 
                    
                    print('lift targeted cohort vs control: %.2f' %liftvscontrol) 
                    print('lift random vs control: %.2f' %liftrandomvscontrol) 
                    print('cpit cohort vs control: %.2f' %cpit) 
                    print('lift targeted-treated vs control: %.2f' %liftpcvscontrol) 
                    #print('cpit cohort: %2f' %cpitcohort) 
            
            ics.append(ic) 
            ### ---- guard against the cost curve going down --- 
            ### for plotting and visualization 
            if len(ios) > 0: 
                if io < ios[-1]: 
                    io = ios[-1] 
            ios.append(io) 
            
            icpus.append(icpu) 
            iopus.append(iopu) 
            treatment_ratios.append(treatment_ratio) 
            
            """ 
            print('decile: ' + str(cnt)) 
            print('cost: ' + str(ic)) 
            print('increment orders: ' + str(io)) 
            print('increment orders per user: ' + str(iopu)) 
            if io != 0: 
                print('cpit: ' + str(ic * 1.0 / io)) 
            """ 
            cnt = cnt + 1 
        
        ## sort the tuple by its index (cost) 
        ics = np.asarray(ics); ics = np.reshape(ics, (-1, 1)) 
        ios = np.asarray(ios); ios = np.reshape(ios, (-1, 1)) 
        ips = np.minimum(ics / ics[-1], 1.0) 
        ios = np.minimum(ios / ios[-1], 1.0) 
        
        if plot_random == True: 
            ips = np.reshape(np.arange(0.0, 1 + 1.0 / num_segments, 1.0 / num_segments), (-1, 1)) 
            ios = np.reshape(np.arange(0.0, 1 + 1.0 / num_segments, 1.0 / num_segments), (-1, 1)) 
        
        ### to guarantee to remove nan issues 
        ips[-1] = 1.0 
        ios[-1] = 1.0 
        
        #ips = ics 
        #ios = ios 
        combined_series = np.concatenate((ics, ios), axis=1) 
        combined_series = np.concatenate((combined_series, ips), axis=1) 
        
        #combined_series = combined_series[combined_series[:, 0].argsort(), :] 
        ## plotting on different colors on same figure 
        ## -- combined_series [:, 0] contains cost 
        ## -- combined_series [:, 1] contains trips/orders 
        ## -- combined_series [:, 2] contains percentages 
        
        ### [for Hong:] feel free to save this to file 
        ### let's define the interface between eng and vis 
        
        print('cmetriccohorts') 
        print(cmetriccohorts) 
        
        cmetriccohorts = np.asarray(cmetriccohorts); cmetriccohorts = np.reshape(cmetriccohorts, (-1, 1)) 
        cmetriccohorts = cmetriccohorts / cmetriccohorts[-1] 
        
        plt.plot(percs, cmetriccohorts, '-o'+color, markersize=12, linewidth=3) 
        plt.xlim(1.0, 0) 
        ### [Todo:] define the file format
        ### ask Bhavya about this evaluator interface
        
        aucc = self.compute_aucc(percs, cmetriccohorts) #ips, ios) 
        plt.xlabel('Percentage of treated') 
        plt.ylabel('Average pre-defined metric (test-set), as % of value at full treatment') 
        
        ax = plt.gca()
        # type(ax)  # matplotlib.axes._subplots.AxesSubplot
        
        # # manipulate to use percentage
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        plt.grid(True)
        
        return aucc, percs, cmetriccohorts 
    
    def AUC_cpit_cost_curve_deciles_cohort_noguardrail(self, pred_values, values, w, n9d_ni_usd, color): 
        ## function plots the cost curve the slope of the curve correspond to 
        ## inverse of CPIT as a growth metric 
        ## [TODO] plan to compute Area Under Curve (AUC) 
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   n9d_ni_usd: list of next-9-day net inflow in usd per user         
        
        ValuesControl = values[w < 0.5] 
        NIControl = n9d_ni_usd[w < 0.5] 
        lenControl = len(ValuesControl) 
        
        rpu_control = np.sum(1.0 * ValuesControl) / lenControl 
        
        nipu_control = np.sum(1.0 * NIControl) / lenControl 
        
        print('rpu_control: ' + str(rpu_control)) 
        print('nipu_control: ' + str(nipu_control))
        
        ValuesFT = values[w > 0.5] 
        NIFT = n9d_ni_usd[w > 0.5] 
        lenFT = len(ValuesFT) 
        
        rpu_ft = np.sum(1.0 * ValuesFT) / lenFT
        
        nipu_ft = np.sum(1.0 * NIFT) / lenFT 
        
        print('rpu_ft: ' + str(rpu_ft)) 
        print('nipu_ft: ' + str(nipu_ft))
        
        ios = [] 
        ics = [] 
        iopus = [] 
        icpus = [] 
        percs = [] ## let's keep percentages of population, for cross checks 
        treatment_ratios = [] 
        
        ## rewriting the code so that we have an inverse rank 
        pred_values = np.reshape(pred_values, (-1, 1)) 
        indices = np.reshape(range(pred_values.shape[0]), (-1, 1)) 
        
        TD = np.concatenate((pred_values, indices), axis = 1) 
        TD = TD[(-1.0 * TD[:, 0]).argsort(), :] 
        #print('printing out the model scores') 
        #print(TD[0:5])
        ## produce the threshold using a percentage (14%) of users to target 
        #threshold_index = int(0.14 * len(TD)) 
        #threshold = TD[threshold_index, 0] 
        
        """ 
        print('the threshold for choosing 14% of users is: ') 
        print(threshold) 
        print('max: ') 
        print(max(TD[:, 0])) 
        print('min: ') 
        print(min(TD[:, 0])) 
        print('average: ') 
        print(np.mean(TD[:, 0]))
        #exit() 
        """ 
        
        num_segments = 10 
        cnt = 0 
        for idx in range(0, TD.shape[0], (int(TD.shape[0]) / int(num_segments))): 
            if idx == 0: 
                io = 0 
                ic = 0 
                iopu = 0 
                icpu = 0 
                num_users = 0 
                treatment_ratio = 0 
            else: 
                selected_user_indices = TD[0:idx, 1] 
                
                model_target_users = np.zeros(w.shape) 
                model_target_users[np.reshape(selected_user_indices, (idx,)).astype(np.int)] = 1 ## index 
                
                treated_targeted_filter = np.logical_and(model_target_users, w) 
                untreated_targeted_filter = np.logical_and(model_target_users, (w < 0.5)) 
                treated_untargeted_filter = np.logical_and(model_target_users < 0.5, w ) 
                untreated_untargeted_filter = np.logical_and(model_target_users < 0.5, (w < 0.5)) 
                
                perc = sum(1.0 * model_target_users) / len(model_target_users) 
                
                treated_target_rpu = sum(1.0 * values[treated_targeted_filter]) / sum(treated_targeted_filter) 
                treated_target_nipu = sum(1.0 * n9d_ni_usd[treated_targeted_filter]) / sum(treated_targeted_filter) 
                
                untreated_target_rpu = sum(1.0 * values[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                untreated_target_nipu = sum(1.0 * n9d_ni_usd[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                
                treated_untarget_rpu = sum(1.0 * values[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                treated_untarget_nipu = sum(1.0 * n9d_ni_usd[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                
                untreated_untarget_rpu = sum(1.0 * values[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                untreated_untarget_nipu = sum(1.0 * n9d_ni_usd[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                
                rpu_cohort = treated_target_rpu * perc + untreated_untarget_rpu * (1 - perc) 
                nipu_cohort = treated_target_nipu * perc + untreated_untarget_nipu * (1 - perc) 
                
                iopu = rpu_cohort - rpu_control
                icpu = -1.0 * (nipu_cohort - nipu_control)
                
                io = iopu * lenControl 
                ic = icpu * lenControl 
                
                cpit = icpu / iopu 
                
                #cpitpc = -1.0 * (treated_target_nipu - nipu_control) / (treated_target_rpu - rpu_control) 
                cpitcohort = -1.0 * (nipu_cohort - nipu_control) / (rpu_cohort - rpu_control) 
                
                liftvscontrol = (rpu_cohort - rpu_control) * 1.0 / (rpu_control) 
                liftrandomvscontrol = (rpu_ft - rpu_control) * 1.0 / (rpu_control) 
                liftpcvscontrol = (treated_target_rpu - rpu_control) * 1.0 / (rpu_control) 
                
                if True: #perc > 0.35 and perc < 0.45: 
                    print('---------------------------->>>>>>') 
                    print('perc - target: %.2f' % perc) 
                    print('treated_target_rpu: %.2f' % treated_target_rpu) 
                    print('treated_target_nipu: %.2f' %treated_target_nipu) 
                    print('nontreated_target_rpu: %.2f' % untreated_target_rpu) 
                    print('nontreated_target_nipu: %.2f' % untreated_target_nipu) 
                    print('treated_nontarget_rpu: %.2f' %treated_untarget_rpu) 
                    print('treated_nontarget_nipu: %.2f' %treated_untarget_nipu) 
                    print('nontreated_nontarget_rpu: %.2f' %untreated_untarget_rpu) 
                    print('nontreated_nontarget_nipu: %.2f' %untreated_untarget_nipu) 
                    
                    cpit_targeted = -1.0 * (treated_target_nipu - untreated_target_nipu) / (treated_target_rpu - untreated_target_rpu) 
                    cpit_nontargeted = -1.0 * (treated_untarget_nipu - untreated_untarget_nipu) / (treated_untarget_rpu - untreated_untarget_rpu) 
                    
                    print('--- with ' + str(perc * 100) + '% targeting, print cpits to treat users and create incrementality in users ---') 
                    print('--> in targeted users: ') 
                    print('cpit = ' + str(cpit_targeted)) 
                    print('--> in non-targeted users: ') 
                    print('cpit = ' + str(cpit_nontargeted)) 
                    
                    print('rpu_control: %.2f' %rpu_control) 
                    print('nipu_control: %.2f' %nipu_control) 
                    print('rpu_ft: %.2f' %rpu_ft) 
                    print('nipu_ft: %.2f' %nipu_ft) 
                    print('rpu_cohort: %.2f' %rpu_cohort) 
                    print('nipu_cohort: %.2f' %nipu_cohort) 
                    
                    print('lift targeted cohort vs control: %.2f' %liftvscontrol) 
                    print('lift random vs control: %.2f' %liftrandomvscontrol) 
                    print('cpit cohort vs control: %.2f' %cpit) 
                    print('lift targeted-treated vs control: %.2f' %liftpcvscontrol) 
                    print('cpit cohort: %2f' %cpitcohort) 
            
            ics.append(ic) 
            ios.append(io) 
            icpus.append(icpu) 
            iopus.append(iopu) 
            treatment_ratios.append(treatment_ratio) 
            
            """
            print('decile: ' + str(cnt)) 
            print('cost: ' + str(ic)) 
            print('increment orders: ' + str(io)) 
            print('increment orders per user: ' + str(iopu))
            if io != 0: 
                print('cpit: ' + str(ic * 1.0 / io)) 
            """
            cnt = cnt + 1 
        
        #print(len(treatment_ratios)) 
        # if False: #color == 'r':
        #     plt.figure()
        #     plt.plot(range(1, 11), treatment_ratios[1:], '-o', markersize=12)
        #     plt.xlabel('decile')
        #     plt.ylabel('treatment percentages')
        #     plt.ylim([0, 0.5])
        #     plt.grid(True)
        #     plt.title('Treatment percentages for top x deciles using targeting model')
        #     plt.show()
        
        ## sort the tuple by its index (cost) 
        ics = np.asarray(ics); ics = np.reshape(ics, (-1, 1)) 
        ios = np.asarray(ios); ios = np.reshape(ios, (-1, 1)) 
        #ips = np.minimum(ics / ics[-1], 1.0) 
        #ios = np.minimum(ios / ios[-1], 1.0) 
        ips = ics / ics[-1] 
        ios = ios / ios[-1]
        combined_series = np.concatenate((ics, ios), axis=1) 
        combined_series = np.concatenate((combined_series, ips), axis=1) 
        
        #combined_series = combined_series[combined_series[:, 0].argsort(), :] 
        ## plotting on different colors on same figure 
        ## -- combined_series [:, 0] contains cost 
        ## -- combined_series [:, 1] contains trips/orders 
        ## -- combined_series [:, 2] contains percentages 
        
        # plt.plot(combined_series[:, 2], combined_series[:, 1], '-o'+color, markersize=10)
        # plt.xlabel('Incremental cost percentage of total inc. cost')
        # plt.ylabel('Incremental trips percentage of total inc. trips(# orders)')
        
        # ax = plt.gca()
        # type(ax)  # matplotlib.axes._subplots.AxesSubplot
        
        # # manipulate to use percentage
        # vals = ax.get_xticks()
        # ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
        # vals = ax.get_yticks()
        # ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
        # plt.grid(True)
    
    def compute_filters_AtBt(self, w, model_target_users): 
        
        # compute filters for At and Bt sets in our documentation 
        # https://docs.google.com/document/d/1c8uCtvh71_Px1PS4NfdF0qDpA-8rZ10GtFYUaGkc6F4/edit 
        # these are sets where 
        #   At: a user is 'selected/targeted' and 'treated' 
        #   Bt: a user is 'selected/targeted' and 'not-treated' 
        # 
        # w: treatment label 
        # model_target_users: the binary vector indicating whether each user is targeted 
        
        target_treated_users = np.logical_and((w > 0.5), (model_target_users > 0.5)) 
        target_nontreated_users = np.logical_and((w <= 0.5), (model_target_users > 0.5)) 
        return target_treated_users, target_nontreated_users 
    
    def compute_increment_orders_target(self, values, w, model_target_users, n9d_ni_usd): 
        
        ## compute incremental orders and incremental cost series 
        ## then plot them to construct a cost curve for selection/targeting 
        ## model evaluation 
        ## values: the value representation or # of orders 
        ## w: the treatment labels 
        ## model_target_users: binary vector indicating whether the user is targeted per dimension 
        ## n9d_ni_usd: the next 9 days inflow numbers for evaluation of incremental cost 
        
        target_treated_users, target_nontreated_users = self.compute_filters_AtBt(w, model_target_users) 
        
        num_users = np.sum(model_target_users) 
        
        values_At = values[target_treated_users] 
        values_Bt = values[target_nontreated_users] 
        
        #values_At = np.minimum(values_At, 20.0) 
        #values_Bt = np.minimum(values_Bt, 20.0) 
        
        nis_At = n9d_ni_usd[target_treated_users] 
        nis_Bt = n9d_ni_usd[target_nontreated_users] 
        
        try: 
            a = 1.0 * np.sum(values_At) 
            b = 1.0 * np.sum(values_Bt) 
            io = a - b 
            
            c = 1.0 * np.sum(nis_At) 
            d = 1.0 * np.sum(nis_Bt) 
            
            # incremental cost: negative of difference in net-inflow 
            ic = -1.0 * ( c - d ) 
            
            # handle divide by zeros 
            a = a / len(values_At) 
            b = b / len(values_Bt) 
            c = c / len(nis_At) 
            d = d / len(nis_Bt) 
            
            ## incremental order per user and its percentage version 
            oput = a 
            opun = b 
            cput = c 
            cpun = d 
            iopu =  a - b 
            icpu =  -1.0 * ( c - d ) 
            
            #num_treated = len(values_At) 
            treatment_ratio = 1.0 * len(values_At) / (len(values_At) + len(values_Bt)) 
            
            num_treated = len(values_At) 
            
            io = iopu * num_treated 
            ic = icpu * num_treated
            
            #io = oput * num_users * treatment_ratio - opun * num_users * (1.0 - treatment_ratio) 
            
            #io = 0.5 * num_users * iopu  
            #ic = 0.5 * num_users * icpu 
        except: 
            io = 0 
            ic = 0 
            iopu = 0 
            icpu = 0 
            treatment_ratio = 0.0 
        
        return io, ic, iopu, icpu, num_users, treatment_ratio
    
    def AUC_ivpu(self, pred_values, values, w, thresh_list, color, n9d_ni_usd, plot=False):
        ## function plots ivpu versus population percentage 
        ## and [TODO] to compute Area Under Curve (AUC) 
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   thresh_list: a range of thresholds generated by numpy.arange based on histogram of pred_values 
        
        percs = [] 
        iopus = [] 
        icpus = [] 
        for t in thresh_list: 
            model_target_users = pred_values > t 
            try: # handle divide by zero 
                perc_users = 1.0 * np.sum(model_target_users) / len(model_target_users) 
            except: 
                perc_users = 0.0 
            d1, d2, iopu, icpu, num_users, treatment_ratio = self.compute_increment_orders_target(values, w, model_target_users, n9d_ni_usd) 
            percs.append(perc_users) 
            iopus.append(iopu) 
            icpus.append(icpu)
        
        # if plot:
        #     ## plotting
        #     plt.plot(percs, iopus, '-o'+color, markersize=12, linewidth=3)
        #     plt.xlabel('% population covered')
        #     plt.ylabel('Incremental # of orders per user')
        #     #plt.ylim([0, 0.3])
        #     plt.grid(True)
        
        return percs, iopus, icpus
        
        """ 
        plt.plot(percs, np.divide(1.0, icpus), color, linewidth=3.0)
        plt.xlabel('% population covered')
        plt.ylabel('Inverse of incremental # of cost per user')
        plt.ylim([0, 5.0])
        plt.grid(True)
        """ 
    
    def AUC_cpit_cost_curve_deciles(self, pred_values, values, w, n9d_ni_usd, color): 
        ## function plots the cost curve whos slopes correspond to 
        ## inverse CPIT aligned with ride-sharing business for promotion measures 
        ## and [TODO] plan to compute Area Under Curve (AUC) 
        ## inputs
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   n9d_ni_usd: list of next-9-day net inflow in usd per user 
        
        ios = [] 
        ics = [] 
        iopus = [] 
        icpus = [] 
        percs = [] ## let's keep percentages of population, for cross checks 
        treatment_ratios = [] 
        
        ## rewriting the code so that we have an inverse rank 
        pred_values = np.reshape(pred_values, (-1, 1)) 
        indices = np.reshape(range(pred_values.shape[0]), (-1, 1)) 
        
        TD = np.concatenate((pred_values, indices), axis = 1) 
        TD = TD[(-1.0 * TD[:, 0]).argsort(), :] 
        
        ## produce the threshold using a percentage (14%) of users to target 
        threshold_index = int(0.14 * len(TD)) 
        threshold = TD[threshold_index, 0] 
        
        """
        print('the threshold for choosing 14% of users is: ') 
        print(threshold) 
        print('max: ') 
        print(max(TD[:, 0]))
        print('min: ')
        print(min(TD[:, 0]))        
        print('average: ') 
        print(np.mean(TD[:, 0]))
        #exit() 
        """ 
        
        num_segments = 10 
        cnt = 0
        for idx in range(0, TD.shape[0], (int(TD.shape[0]) / int(num_segments))): 
            if idx == 0: 
                io = 0
                ic = 0 
                iopu = 0 
                icpu = 0 
                num_users = 0
                treatment_ratio = 0
            else: 
                model_target_users = np.zeros(w.shape) 
                model_target_users[np.reshape(TD[0:idx, 1], (idx,)).astype(np.int)] = 1 ## index 
                io, ic, iopu, icpu, num_users, treatment_ratio = self.compute_increment_orders_target(values, w, model_target_users, n9d_ni_usd) 
            
            #ic = icpu * 0.5 * num_users 
            #io = iopu * 0.5 * num_users 
            ics.append(ic) 
            ios.append(io) 
            icpus.append(icpu) 
            iopus.append(iopu) 
            treatment_ratios.append(treatment_ratio) 
            
            """
            print('decile: ' + str(cnt)) 
            print('cost: ' + str(ic)) 
            print('increment orders: ' + str(io)) 
            print('increment orders per user: ' + str(iopu))
            if io != 0: 
                print('cpit: ' + str(ic * 1.0 / io)) 
            """
            cnt = cnt + 1 
        
        #print(len(treatment_ratios)) 
        # if False: #color == 'r':
        #     plt.figure()
        #     plt.plot(range(1, 11), treatment_ratios[1:], '-o', markersize=12)
        #     plt.xlabel('decile')
        #     plt.ylabel('treatment percentages')
        #     plt.ylim([0, 0.5])
        #     plt.grid(True)
        #     plt.title('Treatment percentages for top x deciles using targeting model')
        #     plt.show()
        
        ## sort the tuple by its index (cost) 
        ics = np.asarray(ics); ics = np.reshape(ics, (-1, 1)) 
        ios = np.asarray(ios); ios = np.reshape(ios, (-1, 1)) 
        combined_series = np.concatenate((ics, ios), axis=1) 
        #combined_series = combined_series[combined_series[:, 0].argsort(), :]
        ## plotting on different colors on same figure
        # plt.plot(combined_series[:, 0], combined_series[:, 1], '-o'+color, markersize=10)
        # plt.xlabel('incremental cost ')
        # plt.ylabel('Incremental trips (# orders)')
        
        # plt.grid(True)
    
    def AUC_cpit_cost_curve(self, pred_values, values, w, thresh_list, n9d_ni_usd, color): 
        ## function plots the cost curve whos slopes correspond to 
        ## inverse CPIT aligned with ride-sharing business for promotion measures 
        ## and [TODO] plan to compute Area Under Curve (AUC) 
        ## inputs
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   thresh_list: a range of thresholds generated by numpy.arange based on histogram of pred_values 
        ##   n9d_ni_usd: list of next-9-day net inflow in usd per user 
        
        ios = [] 
        ics = [] 
        iopus = [] 
        icpus = [] 
        percs = [] ## let's keep percentages of population, for cross checks 
        treatment_percs = [] 
        
        for t in thresh_list: 
            model_target_users = pred_values > t 
            try: # handle divide by zero 
                perc_users = 1.0 * np.sum(model_target_users) / len(model_target_users) 
            except: 
                perc_users = 0.0 
            io, ic, iopu, icpu, num_users, treatment_ratio = self.compute_increment_orders_target(values, w, model_target_users, n9d_ni_usd) 
            #ic = icpu * 0.5 * num_users 
            #io = iopu * 0.5 * num_users 
            ics.append(ic) 
            ios.append(io) 
            icpus.append(icpu) 
            iopus.append(iopu) 
            percs.append(perc_users) 
            treatment_percs.append(treatment_ratio)         
        
        ## sort the tuple by its index (cost) 
        
        ics = np.asarray(ics); ics = np.reshape(ics, (-1, 1)) 
        ios = np.asarray(ios); ios = np.reshape(ios, (-1, 1)) 
        combined_series = np.concatenate((ics, ios), axis=1) 
        #combined_series = combined_series[combined_series[:, 0].argsort(), :]
        ## plotting on different colors on same figure 
        # plt.plot(combined_series[:, 0], combined_series[:, 1], color)
        # plt.xlabel('incremental cost ')
        # plt.ylabel('Incremental trips (# orders)')
        
        # plt.grid(True)
        
    def Aggregate_CF_Curves_Best_std(self, cf_o_score_list, cf_c_score_list, cf_d_score_list, d2dlamb, wt, ot, ct): 
        import pandas as pd 
        best_cf_aucc = -100.0 
        best_cf_aumc = -100.0 
        ## temp  solution: load d2dt from csv in causal forest test data 
        import pandas as pd 
        dtable = pd.read_csv('causaltree/causal_forest_r_data_uscensus_pub_causal_data_intensity_matching_d2dist_test.csv') 
        d2dt = dtable['d2d'].as_matrix() 
        cfaucclist = [] 
        cfaumclist = [] 
        for i in range(len(cf_o_score_list)): 
            ### this section is to load the results trained by grf R code ### 
            ot_cf = pd.read_csv(cf_o_score_list[i]) 
            ct_cf = pd.read_csv(cf_c_score_list[i]) 
            dt_cf = pd.read_csv(cf_d_score_list[i]) 
            
            ot_cf = ot_cf.as_matrix() 
            Ocfscores = ot_cf[0][1:] 
            
            ct_cf = ct_cf.as_matrix() 
            Ccfscores = ct_cf[0][1:] 
            
            dt_cf = dt_cf.as_matrix() 
            Dcfscores = dt_cf[0][1:] 
            
            #cfscore = np.divide(Ocfscores, Ccfscores) 
            #cfscore = Ocfscores - 10.0 * Ccfscores
            #cfscore = -1.0 * (np.divide(np.maximum(Ccfscores, 1e-7), np.maximum(Ocfscores, 1e-7)) + d2dlamb*Dcfscores) 
            #cfscore = -1.0 * (np.divide(np.maximum(Ccfscores, 1e-7), np.maximum(Ocfscores, 1e-7)) + d2dlamb*Dcfscores) 
            cfscore = np.multiply(Dcfscores, (Ocfscores - d2dlamb * Ccfscores)) 
            
            cfaucc, cfcs, cfos = self.AUC_cpit_cost_curve_deciles_cohort(cfscore, ot, wt, -1.0 * ct, 'g') # causal forest aucc and plotting e 
            cfaumc, cfps, cfms = self.AUC_cpit_cost_curve_deciles_cohort_d2dist(cfscore, ot, wt, -1.0 * ct, d2dt, d2dlamb, 'g') 
            
            #cfcs = np.reshape(cfcs, (-1, 1)) 
            cfos = np.reshape(cfos, (-1, 1)) 
            #cfps = np.reshape(cfps, (-1, 1)) 
            cfms = np.reshape(cfms, (-1, 1)) 

            cfms = cfms / cfms[-1] 
            if i == 0: 
                cfoslist = cfos 
                cfmslist = cfms 
            else: 
                cfoslist = np.concatenate((cfoslist, cfos), axis = 1)
                cfmslist = np.concatenate((cfmslist, cfms), axis = 1) 
            
            cfaucclist.append(cfaucc) 
            cfaumclist.append(cfaumc) 
            
            ## aucc 
            if cfaucc > best_cf_aucc:
                best_cf_aucc = cfaucc
                best_cf_cc_series_cs = cfcs
                best_cf_cc_series_os = cfos
            ## aumc 
            if cfaumc > best_cf_aumc:
                best_cf_aumc = cfaumc
                best_cf_cc_series_ps = cfps
                best_cf_cc_series_ms = cfms
        cfos_std = np.std(cfoslist, axis = 1)
        cfms_std = np.std(cfmslist, axis = 1) 
        std_cf_aucc = np.std(np.asarray(cfaucclist)) 
        std_cf_aumc = np.std(np.asarray(cfaumclist)) 
        return best_cf_aucc, std_cf_aucc, best_cf_cc_series_cs, best_cf_cc_series_os, best_cf_aumc, std_cf_aumc, best_cf_cc_series_ps, best_cf_cc_series_ms, cfos_std, cfms_std
    
    def Aggregate_Curves_Best_std(self, filelist): 
        
        best_ran_aucc = -100.0 
        best_quasi_aucc = -100.0 
        best_ctpm_aucc = -100.0 
        best_drm_aucc = -100.0 
        
        best_ran_aumc = -100.0 
        best_quasi_aumc = -100.0 
        best_ctpm_aumc = -100.0 
        best_drm_aumc = -100.0 
        
        ranaucclist = [] 
        ranaumclist = [] 
        quasiaucclist = [] 
        quasiaumclist = [] 
        ctpmaucclist = [] 
        ctpmaumclist = [] 
        drmaucclist = [] 
        drmaumclist = [] 
        
        for i in range(len(filelist)): 
            D = pkl.load(open(filelist[i], 'rb')) 
            D['rancs'] = np.reshape(D['rancs'], (-1, 1)) 
            D['ranos'] = np.reshape(D['ranos'], (-1, 1)) 
            D['ranps'] = np.reshape(D['ranps'], (-1, 1)) 
            D['ranms'] = np.reshape(D['ranms'], (-1, 1)) 
            D['ranms'] = D['ranms'] / D['ranms'][-1]
            
            D['quasics'] = np.reshape(D['quasics'], (-1, 1)) 
            D['quasios'] = np.reshape(D['quasios'], (-1, 1)) 
            D['quasips'] = np.reshape(D['quasips'], (-1, 1)) 
            D['quasims'] = np.reshape(D['quasims'], (-1, 1))
            D['quasims'] = D['quasims'] / D['quasims'][-1]
            
            D['ctpmcs'] = np.reshape(D['ctpmcs'], (-1, 1)) 
            D['ctpmos'] = np.reshape(D['ctpmos'], (-1, 1))  
            D['ctpmps'] = np.reshape(D['ctpmps'], (-1, 1)) 
            D['ctpmms'] = np.reshape(D['ctpmms'], (-1, 1))  
            D['ctpmms'] = D['ctpmms'] / D['ctpmms'][-1]
            
            D['drmcs'] = np.reshape(D['drmcs'], (-1, 1))  
            D['drmos'] = np.reshape(D['drmos'], (-1, 1))  
            D['drmps'] = np.reshape(D['drmps'], (-1, 1))  
            D['drmms'] = np.reshape(D['drmms'], (-1, 1)) 
            D['drmms'] = D['drmms'] / D['drmms'][-1]
            
            if i == 0: 
                rancslist = np.reshape(D['rancs'], (-1, 1)) 
                ranoslist = np.reshape(D['ranos'], (-1, 1)) 
                ranpslist = np.reshape(D['ranps'], (-1, 1)) 
                ranmslist = np.reshape(D['ranms'], (-1, 1)) 
                
                quasicslist = np.reshape(D['quasics'], (-1, 1)) 
                quasioslist = np.reshape(D['quasios'], (-1, 1)) 
                quasipslist = np.reshape(D['quasips'], (-1, 1)) 
                quasimslist = np.reshape(D['quasims'], (-1, 1))
                
                ctpmcslist = np.reshape(D['ctpmcs'], (-1, 1)) 
                ctpmoslist = np.reshape(D['ctpmos'], (-1, 1))  
                ctpmpslist = np.reshape(D['ctpmps'], (-1, 1)) 
                ctpmmslist = np.reshape(D['ctpmms'], (-1, 1))  
                
                drmcslist = np.reshape(D['drmcs'], (-1, 1))  
                drmoslist = np.reshape(D['drmos'], (-1, 1))  
                drmpslist = np.reshape(D['drmps'], (-1, 1))  
                drmmslist = np.reshape(D['drmms'], (-1, 1))  
            
            else:
                rancslist = np.concatenate((rancslist, D['rancs']), axis = 1) 
                ranoslist = np.concatenate((ranoslist, D['ranos']), axis = 1) 
                ranpslist = np.concatenate((ranpslist, D['ranps']), axis = 1) 
                ranmslist = np.concatenate((ranmslist, D['ranms']), axis = 1) 
                
                quasicslist = np.concatenate((quasicslist, D['quasics']), axis = 1) 
                quasioslist = np.concatenate((quasioslist, D['quasios']), axis = 1) 
                quasipslist = np.concatenate((quasipslist, D['quasips']), axis = 1) 
                quasimslist = np.concatenate((quasimslist, D['quasims']), axis = 1) 
                
                ctpmcslist = np.concatenate((ctpmcslist, D['ctpmcs']), axis = 1) 
                ctpmoslist = np.concatenate((ctpmoslist, D['ctpmos']), axis = 1) 
                ctpmpslist = np.concatenate((ctpmpslist, D['ctpmps']), axis = 1) 
                ctpmmslist = np.concatenate((ctpmmslist, D['ctpmms']), axis = 1) 
                
                drmcslist = np.concatenate((drmcslist, D['drmcs']), axis = 1) 
                drmoslist = np.concatenate((drmoslist, D['drmos']), axis = 1) 
                drmpslist = np.concatenate((drmpslist, D['drmps']), axis = 1) 
                drmmslist = np.concatenate((drmmslist, D['drmms']), axis = 1) 
            
            ## aucc 
            if D['ranaucc'] > best_ran_aucc: 
                best_ran_aucc = D['ranaucc'] 
                best_ran_cc_series_cs = D['rancs'] 
                best_ran_cc_series_os = D['ranos'] 
            ranaucclist.append(D['ranaucc'][0])
            
            if D['quasiaucc'] > best_quasi_aucc: 
                best_quasi_aucc = D['quasiaucc'] 
                best_quasi_cc_series_cs = D['quasics'] 
                best_quasi_cc_series_os = D['quasios'] 
            quasiaucclist.append(D['quasiaucc'][0]) 
            
            if D['ctpmaucc'] > best_ctpm_aucc: 
                best_ctpm_aucc = D['ctpmaucc'] 
                best_ctpm_cc_series_cs = D['ctpmcs'] 
                best_ctpm_cc_series_os = D['ctpmos'] 
            ctpmaucclist.append(D['ctpmaucc'][0]) 
            
            if D['drmaucc'] > best_drm_aucc: 
                best_drm_aucc = D['drmaucc'] 
                best_drm_cc_series_cs = D['drmcs'] 
                best_drm_cc_series_os = D['drmos'] 
            drmaucclist.append(D['drmaucc'][0]) 
            
            ## aumc 
            if D['ranaumc'] > best_ran_aumc: 
                best_ran_aumc = D['ranaumc'] 
                best_ran_cc_series_ps = D['ranps'] 
                best_ran_cc_series_ms = D['ranms'] 
            ranaumclist.append(D['ranaumc'][0]) 
            
            if D['quasiaumc'] > best_quasi_aumc: 
                best_quasi_aumc = D['quasiaumc'] 
                best_quasi_cc_series_ps = D['quasips'] 
                best_quasi_cc_series_ms = D['quasims'] 
            quasiaumclist.append(D['quasiaumc'][0]) 
            
            if D['ctpmaumc'] > best_ctpm_aumc: 
                best_ctpm_aumc = D['ctpmaumc'] 
                best_ctpm_cc_series_ps = D['ctpmps'] 
                best_ctpm_cc_series_ms = D['ctpmms'] 
            ctpmaumclist.append(D['ctpmaumc'][0]) 
            
            if D['drmaumc'] > best_drm_aumc: 
                best_drm_aumc = D['drmaumc'] 
                best_drm_cc_series_ps = D['drmps'] 
                best_drm_cc_series_ms = D['drmms'] 
            drmaumclist.append(D['drmaumc'][0]) 
        
        rancs_std = np.std(rancslist, axis = 1) 
        ranos_std = np.std(ranoslist, axis = 1)             
        ranms_std = np.std(ranmslist, axis = 1) 
        
        quasics_std = np.std(quasicslist, axis = 1)             
        quasios_std = np.std(quasioslist, axis = 1)             
        quasims_std = np.std(quasimslist, axis = 1) 
        
        ctpmcs_std = np.std(ctpmcslist, axis = 1)             
        ctpmos_std = np.std(ctpmoslist, axis = 1)             
        ctpmms_std = np.std(ctpmmslist, axis = 1) 
        
        drmcs_std = np.std(drmcslist, axis = 1) 
        drmos_std = np.std(drmoslist, axis = 1) 
        drmms_std = np.std(drmmslist, axis = 1) 
        
        std_ran_aucc = np.std(np.asarray(ranaucclist))
        std_quasi_aucc = np.std(np.asarray(quasiaucclist)) 
        std_ctpm_aucc = np.std(np.asarray(ctpmaucclist)) 
        std_drm_aucc = np.std(np.asarray(drmaucclist)) 
        
        std_ran_aumc = np.std(np.asarray(ranaumclist)) 
        std_quasi_aumc = np.std(np.asarray(quasiaumclist)) 
        std_ctpm_aumc = np.std(np.asarray(ctpmaumclist)) 
        std_drm_aumc = np.std(np.asarray(drmaumclist)) 
        
        return_list = [best_ran_aucc, 
                       std_ran_aucc, 
                       best_ran_cc_series_cs, 
                       best_ran_cc_series_os, 
                       best_ran_aumc, 
                       std_ran_aumc, 
                       best_ran_cc_series_ps, 
                       best_ran_cc_series_ms,                
                       best_quasi_aucc, 
                       std_quasi_aucc, 
                       best_quasi_cc_series_cs, 
                       best_quasi_cc_series_os, 
                       best_quasi_aumc, 
                       std_quasi_aumc, 
                       best_quasi_cc_series_ps, 
                       best_quasi_cc_series_ms,                
                       best_ctpm_aucc,
                       std_ctpm_aucc, 
                       best_ctpm_cc_series_cs, 
                       best_ctpm_cc_series_os, 
                       best_ctpm_aumc, 
                       std_ctpm_aumc, 
                       best_ctpm_cc_series_ps, 
                       best_ctpm_cc_series_ms,                
                       best_drm_aucc, 
                       std_drm_aucc, 
                       best_drm_cc_series_cs, 
                       best_drm_cc_series_os, 
                       best_drm_aumc, 
                       std_drm_aumc, 
                       best_drm_cc_series_ps, 
                       best_drm_cc_series_ms,                
                       ranos_std, 
                       ranms_std,                
                       quasios_std, 
                       quasims_std,                
                       ctpmos_std, 
                       ctpmms_std,                
                       drmos_std, 
                       drmms_std
                       ] 
        return return_list
