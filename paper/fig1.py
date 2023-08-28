"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import cv2
import matplotlib
import matplotlib.pyplot as plt
from fig_utils import *
from scipy.stats import wilcoxon

from facemap import keypoints


def panel_percentile_error(ax, data_path):
    errors = np.load(f'{data_path}net_results/facemap_benchmark_distances.npy', allow_pickle=True).item()['test_distances']
    keypoint_labels = np.array(np.load(f'{data_path}net_results/facemap_benchmark_distances.npy', allow_pickle=True).item()['bodyparts'])
    igood = np.logical_and(keypoint_labels != 'paw', keypoint_labels != 'nosebridge')
    errors = errors[:,igood]
    keypoint_labels = keypoint_labels[igood]
    dk = np.load(f'{data_path}net_results/example_labels_frame.npy', allow_pickle=True).item()
    frame = dk['frame'][:,:,0]
    kps = dk['kps'].reshape(-1,2)[igood]
    
    lw=0.5
    x_pos, y_pos = kps[:,0], kps[:,1]
    keypoint_radius50 = np.nanpercentile(errors, 50, axis=0)
    keypoint_radius75 = np.nanpercentile(errors, 75, axis=0)
    keypoint_radius90 = np.nanpercentile(errors, 90, axis=0)
    
    ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
    ax.plot([10,30], [10,10], color='w')
    ax.text(10,15, '20 px', va='top', ha='left', fontsize='small', color='w')
    for j in range(len(x_pos)):
        if keypoint_labels[j]!="whisker(c2)":
            ji = kp_labels_old.index(keypoint_labels[j])
        else:
            ji = 5
        if np.isnan(x_pos[j]):
            continue
        c = plt.Circle(xy=(x_pos[j], y_pos[j]), radius=keypoint_radius50[j],
                    color=kp_colors[ji], fill=None, linestyle='-', linewidth=lw)
        ax.add_patch(c)
        c = plt.Circle(xy=(x_pos[j], y_pos[j]), radius=keypoint_radius75[j],
                    color=kp_colors[ji], fill=None, linestyle='dashdot', linewidth=lw)
        ax.add_patch(c)
        c = plt.Circle(xy=(x_pos[j], y_pos[j]), radius=keypoint_radius90[j],
                    color=kp_colors[ji], fill=None, linestyle='dotted', linewidth=lw)
        ax.add_patch(c)
    ax.set_title('error percentile')
    ax.set_xlim(0, frame.shape[-2])
    ax.set_ylim(frame.shape[-1], 0)
    ax.axis('off')

    percentile50_legend = plt.Line2D([], [], color="k",
                                        ls='-', ms=5, fillstyle='none',
                                    linewidth=1, label='50th')
    percentile75_legend = plt.Line2D([], [], color='k', 
                                        ls='dashdot', ms=5, fillstyle='none',
                                    linewidth=1, label='75th')
    percentile90_legend = plt.Line2D([], [], color='k', 
                                        ls='dotted', ms=5, fillstyle='none',
                                    linewidth=1, label='90th')
    legend = ax.legend( frameon=False,labelcolor='k', 
                title_fontsize=10, fontsize='small', ncol=3,
                handles=[percentile50_legend, percentile75_legend, percentile90_legend],
                loc=(-.25,-0.2))

def panel_benchmark_error(ax, data_path):
    # Plot avg. error for grouped bodyparts
    # Make a scatter plot of the distances for each keypoint
    dat = np.load(f'{data_path}net_results/facemap_benchmark_distances.npy', allow_pickle=True).item()
    bodyparts = np.array(dat['bodyparts'])
    bodyparts_groups = ['eye', 'mouth', 'nose', 'whiskers']
    sub_groups = [['eye(back)', 'eye(bottom)', 'eye(front)', 'eye(top)'],
                ['lowerlip', 'mouth'],
                ['nose(bottom)', 'nose(r)', 'nose(tip)', 'nose(top)'],# 'nosebridge'],
                ['whisker(c1)', 'whisker(c2)', 'whisker(d1)'],
                ['paw'],]
    order = ["test", "human"]
    markers = ['x', 'o']
    markersize = 5
    
    for k, name in enumerate(order): 
        distances = dat[f'{name}_distances']
        marker = markers[k]
        for group_idx, group in enumerate(bodyparts_groups):
            selected_bodyparts_idx = []
            for i, bodypart in enumerate(bodyparts):
                try:
                    found = sub_groups[group_idx].index(bodypart)
                    selected_bodyparts_idx.append(bodyparts.tolist().index(bodypart))
                except ValueError:
                    continue   
            print(group) 
            mean_distance_group = np.nanmean(distances[:,selected_bodyparts_idx])
            confidence_interval_95 = get_confidence_interval(distances[:,selected_bodyparts_idx].flatten(), alpha=1)
            err_low = mean_distance_group - confidence_interval_95[0]
            err_high = confidence_interval_95[1] - mean_distance_group
            ax.errorbar(group_idx, mean_distance_group, yerr=[[err_low], [err_high]], marker=marker, c='k', ms=markersize, fillstyle='none')
        # Set x ticks and labels
        ax.set_xticks(np.arange(len(bodyparts_groups)))
        ax.set_xticklabels(["{}".format(i) for i in bodyparts_groups], rotation=90)
        # Change fontsize of y-axis label
    ax.set_yticklabels(ax.get_yticks())
    plt.ylabel("error (px)")
    test_legend = plt.Line2D([], [], color='k', marker='x', label='Facemap', ms=markersize, linestyle="None", fillstyle='none')
    human_legend = plt.Line2D([], [], color='k', marker='o', label='Human', ms=markersize, linestyle="None", fillstyle='none')
    legend = ax.legend(facecolor='w', labelcolor='k', 
            edgecolor='k', fontsize=8, ncol=1,
            handles=[test_legend, human_legend],
            loc=(0.05,0.85))
    ax.set_ylim([0,9.])

def panel_error_speed_comparison(ax, data_path):
    models = ['facemap', 'dlc_resnet50', 'dlc_mobilenet', 'sleap_n16']#, 'sleap_n32']
    #colors = ['k','k','k','k']#, np.ones(3)*0.25, np.ones(3)*0.7, np.ones(3)*0.5]#'darkgreen', 'limegreen', 'purple']#, 'mediumorchid']
    colors = ['k', 'darkgreen', 'limegreen', 'purple']#, 'mediumorchid']
    labels = ['Facemap', 'DeepLabCut\nResNet50', 'DeepLabCut\nMobilenet', 'SLEAP\n(default)']#, 'SLEAP\nn=32']
    adjust_poss = [(40,0.05), (0,0.04), (40,0.1), (35,0.19)]#, (25,0.25)]
    d = np.load(f'{data_path}net_results/facemap_benchmark_distances.npy', allow_pickle=True).item()
    keypoint_labels = np.array(d['bodyparts'])
    errors_fm = d['test_distances']
    for model, model_label, c, adjust_pos in zip(models, labels, colors, adjust_poss):
        if model!='facemap':
            errors = np.load(f'{data_path}net_results/{model}.npy', allow_pickle=True).item()['dist_error']
        else:
            errors = errors_fm.copy()
        inference_speeds = np.load(f'{data_path}net_results/{model}_v100_inference_speed_batchsize1.npy')
        inference_speed = np.nanmean(inference_speeds)
        # remove paw
        igood = np.logical_and(keypoint_labels != 'paw', keypoint_labels != 'nosebridge')
        errors = errors[:,igood]
        model_error = np.nanmean(errors)
        print(model_label, model_error)
        confidence_interval_95 = get_confidence_interval(errors, alpha=1)
        err_low = model_error - confidence_interval_95[0]
        err_high = confidence_interval_95[1] - model_error
        #err_low = np.nanpercentile(errors, 5)
        #err_high = np.nanpercentile(errors, 95)
        model_err = [[err_low], [err_high]]
        speed_ci = get_confidence_interval(inference_speeds, alpha=1)
        speed_err_low = inference_speed - speed_ci[0]
        speed_err_high = speed_ci[1] - inference_speed
        speed_err = [[speed_err_low], [speed_err_high]]
        ax.errorbar(inference_speed, model_error, marker="o", xerr=speed_err, yerr=model_err,  ms=5, color=c)
        # Add annotation for model_name
        xpos = inference_speed
        ypos = model_error + model_err[1][0] + 0.02
        w="normal"
        ax.text(xpos+(model=='sleap_n16')*20, ypos, model_label,
                 ha='center', color=c)
        #ax.annotate(model_label, xy=(xpos, ypos), color=c,
        #            xytext=(xpos+adjust_pos[0], ypos+adjust_pos[1]),
        #            ha="center", fontname="Arial")
    ax.set_xlabel("processing speed (FPS)\n(batch size = 1)", fontsize=12)
    ax.set_ylabel("average error (px)", fontsize=12)
    ax.set_xlim(0, 400)
    ax.set_ylim(3.7, 6.)
    ax.set_yticks(np.arange(4,6.5,0.5))
    #ax.tick_params(axis='both', which='major', labelsize=10)

def panel_varexp_future(ax, varexps_areas, tlags):
    areas = ['eye', 'whisker', 'nose']
    cols = kp_colors[[2,13,7]]
    vea_mean = np.array(varexps_areas).mean(axis=0) * 100
    vea_ste = np.array(varexps_areas).std(axis=0) / (len(varexps_areas)-1)**0.5 * 100
    for k in range(3):
        ax.plot(tlags / 50, vea_mean[k], color=cols[k])
        ax.fill_between(tlags / 50, vea_mean[k]-vea_ste[k], vea_mean[k]+vea_ste[k], color=cols[k], alpha=0.25)
        ax.text(1.0, 0.7-k*0.1, areas[k], transform=ax.transAxes, color=cols[k], ha='right')
    ax.set_xlim([0,20])
    #ax.set_xscale('log')
    ax.set_ylim([0, 100])
    ax.set_ylabel('variance explained (%)')
    ax.set_xlabel('time in future (s)')

def panel_future_decay(ax, varexps_areas, tlags):
    areas = ['eye', 'whisker', 'nose']
    cols = kp_colors[[2,13,7]]
    hm = np.zeros((3, len(varexps_areas)))
    for k in range(3):
        vk = np.array(varexps_areas)[:,k]
        vd = vk < vk[:,[0]]*0.5
        for j in range(len(vd)):
            vj = np.nonzero(vd[j])[0]
            hm[k,j] = tlags[vj[0]]/50. if len(vj)>0 else tlags[-1]/50.
    print('timescales for')
    print(areas)
    print(hm.mean(axis=1))
    for i in range(3):
        for j in range(i+1,3):
            print(wilcoxon(hm[i], hm[j]))
    ymax=60
    dy = [-20, -30, 0]
    dx = [0.5, 1.5, 1.]
    xh = np.arange(0,3)[:,np.newaxis] + np.random.randn(*hm.shape)*0.015
    print(xh.shape)
    for k in range(3):
        ax.scatter(xh[k], hm[k], s=5, alpha=1, color=cols[k])
        ax.scatter(k, hm[k].mean(), marker='x', s=40, color='k')#color=cols[k])
        ax.plot([k, (k+1)%3],(ymax+dy[k])*np.ones(2), color='k', lw=1, marker=3, markersize=3)
        ax.text(dx[k], ymax+dy[k], '***', ha='center', va='center')
    ax.plot(xh, hm, color=[0.5,0.5,0.5], zorder=-10, lw=0.25)
    ax.set_ylim([0.1, ymax])
    ax.set_yscale('log')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(areas)
    ax.set_yticks(10.**np.arange(-1,2))
    ax.set_yticklabels(['0.1', '1', '10'])
    ax.set_ylabel('time to half varexp (s)')

def panel_ex_future_pred(ax, x, ypred, keypoint_labels):
    areas = ['eye', 'whisker', 'nose']
    t = 1000
    for k in range(len(areas)):
        ax[k].axis('off')
        pos = ax[k].get_position()
        pos = [pos.x0, pos.y0, pos.width, pos.height]
        ax[k] = ax[k].figure.add_axes([pos[0], pos[1]-.07, pos[2]+0.0, pos[3]+0.07])
        ax[k].axis('off')        
        #ax[k].set_title(areas[k])
    ax[0].set_title('future prediction')
    nk = [0,0,0]
    dy = [3,5,5]
    iperm = np.array([3,0,2,1,5,6,4,7,8,9,10])
    keypoint_labels = np.array(keypoint_labels)[iperm]
    iperm = 2*iperm[:,np.newaxis] + np.arange(0,2)
    iperm = iperm.flatten()
    ypred = ypred[iperm]
    x = x[:,iperm]
    #ypred = 
    for j in range(ypred.shape[0]):
        kp = keypoint_labels[j//2]
        k = [k for k in range(len(areas)) if areas[k] in kp]
        if len(k) > 0:
            k = k[0]
            ji = kp_labels_old.index(kp)
            ax[k].plot(np.arange(0,500*2), nk[k]*dy[k] + x[:,2*(j//2)+(j+1)%2][t-500:t+500],
                    linestyle='-', zorder=-10, color=kp_colors[ji], lw=1)
            ax[k].plot(np.arange(0,500)+500, nk[k]*dy[k] + ypred[2*(j//2)+(j+1)%2,t], '-',
                        zorder=10, lw=3, color=[.5,.5,.5])
            ax[k].text(-100,nk[k]*dy[k], ['x','y'][(j+1)%2], color=kp_colors[ji])
            if nk[k]==0 and k==0:
                ax[0].text(1100,-4, 'keypoint', color=kp_colors[ji], ha='right')
                ax[0].text(1100,-6.5, 'prediction', color=[.5,.5,.5], ha='right')
            nk[k] += 1.2 if j%2==0 else 1.7
    for k in range(3):
        ax[k].plot([0,250], [-4,-4], color='k')
    ax[0].text(-20,-7, '5 sec.')


def panel_ex_behaviors(grid0, data_path, db, il):
    mname, datexp, blk = db['mname'], db['datexp'], db['blk']
    cam_filename = f'{data_path}cam/cam1_{mname}_{datexp}_{blk}.avi'
    kp_path0 = f'{data_path}keypoints/cam1_{mname}_{datexp}_{blk}_FacemapPose.h5'
    xy, keypoint_labels = keypoints.load_keypoints(kp_path0, outlier_filter=False, keypoint_labels=None,
                                                    confidence_threshold=False)
    # remove paw 
    xy = xy[:,:-1]
    keypoint_labels = keypoint_labels[:-1]
    behaviors = ['blinking', 'whisking', 'sniffing']
    time_points = [188370, 188273, 95940] # 183900 # 332394
    t0,t1 = [84,10,55], [66,83,147]
    twin = 200 
    selected_keypoints = [['eye(top)','eye(back)', 'eye(front)', 'eye(bottom)'], 
                        ['whisker(c1)', 'whisker(d1)', 'whisker(d2)'][::-1], 
                        ['nose(top)', 'nose(tip)', 'nose(bottom)'][::-1]]
    dy = [30, 50, 40]
    y = [20, 20, 20]
    for j in range(3):
        for k in range(2):
            ax = plt.subplot(grid0[j,k])
            if k==0:
                il = plot_label(ltr, il, ax,
                        mtransforms.ScaledTranslation(-20/72, 10/72, grid0.figure.dpi_scale_trans), fs_title)
            pos = ax.get_position()
            pos = [pos.x0, pos.y0, pos.width, pos.height]
            t = time_points[j]
            if k==0:
                capture = cv2.VideoCapture(cam_filename)
                capture.set(cv2.CAP_PROP_POS_FRAMES, t0[j]+t)
                ret, frame = capture.read()
                img = frame[115:,30:,0].astype('float32') / 255.
                img = np.clip(img*2.4, 0, 1.)
                xpad = 50
                imga = np.ones((img.shape[0], img.shape[1]*2+xpad))
                imga[:,:img.shape[1]] = img
                capture.set(cv2.CAP_PROP_POS_FRAMES, t1[j]+t)
                ret, frame = capture.read()
                img = frame[115:,30:,0].astype('float32') / 255.
                img = np.clip(img*2.4, 0, 1.)
                capture.release()
                imga[:,img.shape[1]+xpad:] = img
                ax.imshow(imga, cmap='gray')
                ax.axis('off')
                for i,kp in enumerate(keypoint_labels):
                    kl = keypoint_labels.index(kp)
                    ji = kp_labels_old.index(keypoint_labels[i])
                    xyp = xy[t0[j]+t,i].copy()
                    ax.scatter(xyp[0]-30, xyp[1]-115, color=kp_colors[ji], s=8)
                    xyp = xy[t1[j]+t,i].copy()
                    ax.scatter(xyp[0]-30 + img.shape[1] + xpad, xyp[1]-115, color=kp_colors[ji], s=8)
                ax.set_title(behaviors[j], fontsize='medium')                    
            else:
                ax.set_position([pos[0]-0.03, pos[1]-0.01, pos[2]+0.04, pos[3]+0.015])
                for i,kp in enumerate(selected_keypoints[j]):
                    kl = keypoint_labels.index(kp)
                    ji = kp_labels_old.index(kp)
                    xyp = xy[t:t+twin,kl].copy()
                    xyp -= xyp.mean(axis=0)
                    ax.plot(xyp[:,0] + i*dy[j], color=kp_colors[ji], lw=1)
                    if j!=1:
                        ax.plot(-1*xyp[:,1] + i*dy[j], color=kp_colors[ji], linestyle='--', lw=1)
                    if kp[:7]=='whisker':
                        kp = 'whisker(' + ['I','II','III'][2-i] + ')'
                    ax.text(0,i*dy[j]+dy[j]/2,kp, color=kp_colors[ji], va='center')
                ax.set_xlim([0,twin])
                #print(ax.get_ylim())
                ax.plot([0, 25], [-y[j],-y[j]], color='k')
                #ax.plot([-5,-5], [-y[j],-y[j]+25], color='k')
                #if j==2:
                ax.text(0, -y[j]*1.2, '0.5 sec.', va='top')
                if j==0:
                    h0,=ax.plot(-100,0,'k-')
                    h1,=ax.plot(-100,0,'k--')
                    ax.legend([h0, h1], ['x keypoint', 'y keypoint'], loc=[0.6, 1], frameon=False)
                    ax.text(0, 1.2, 'Keypoints', transform=ax.transAxes, fontsize='large')
            ax.axis('off')
    return il 

def fig1(data_path, db, save_fig=False):
    # load data for future prediction figures
    # (other data loaded in panel-specific functions)
    ved = np.load(f'{data_path}proc/keypoints/varexp_future_kpfilt.npy', allow_pickle=True).item()
    
    varexps_areas = ved['varexps_areas']
    tlags = ved['tlags']
    x_ex = ved['x_ex']
    ypred_ex = ved['ypred_ex']
    keypoint_labels_pred = ved['keypoint_labels']

    fig = plt.figure(figsize=(14,7.8))
    trans = mtransforms.ScaledTranslation(-40/72, 7/72, fig.dpi_scale_trans)
    grid = plt.GridSpec(3,6, figure=fig, left=0.05, right=0.99, top=0.95, bottom=0.08, 
    wspace = 0.75, hspace = 0.5)

    il=0
    ax = plt.subplot(grid[0, :2])
    pos = ax.get_position()
    ax.axis('off')
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = fig.add_axes([pos[0]-0.14, pos[1]-.08, pos[2]+0.13, pos[3]+0.13])
    ax.imshow(plt.imread(f'{data_path}figs/mouse_face_keypoints.png'))
    ax.axis('off')
    ax = fig.add_axes([pos[0], pos[1], pos[2], pos[3]])
    il = plot_label(ltr, il, ax, trans, fs_title)
    ax.axis('off')

    ax = plt.subplot(grid[0, 2:4])
    pos = ax.get_position()
    ax.axis('off')
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = fig.add_axes([pos[0]-0.16, pos[1]-0.06, pos[2]+0.12   , pos[3]+0.12])
    ax.axis('off')
    ax.imshow(plt.imread(f'{data_path}figs/facemap_network.png'))
    ax = fig.add_axes([pos[0]-0.13, pos[1], pos[2], pos[3]])
    il = plot_label(ltr, il, ax, trans, fs_title)
    ax.axis('off')

    
    ax = plt.subplot(grid[0, 4])
    pos = ax.get_position()
    ax.axis('off')
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = fig.add_axes([pos[0]-0.05, pos[1], pos[2]+0.02, pos[3]+0.02])
    ax.axis('off')
    il = plot_label(ltr, il, ax, trans, fs_title)
    panel_percentile_error(ax, data_path)
    
    ax = plt.subplot(grid[0, -1])
    il = plot_label(ltr, il, ax, trans, fs_title)
    panel_benchmark_error(ax, data_path)

    ax = plt.subplot(grid[1:, :1])
    il = plot_label(ltr, il, ax, trans, fs_title)
    panel_error_speed_comparison(ax, data_path)

    grid0 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=grid[1:, 1:4], hspace=0.65)
    il = panel_ex_behaviors(grid0, data_path, db, il)
    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[1:-1, 4:], 
                                                        wspace=0.1, hspace=0)
    ax = [plt.subplot(grid1[0,k]) for k in range(3)]
    il = plot_label(ltr, il, ax[0], trans, fs_title)
    panel_ex_future_pred(ax, x_ex, ypred_ex, keypoint_labels_pred)

    grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[-1, 4:], 
                                                        wspace=0.5, hspace=0)
    trans = mtransforms.ScaledTranslation(-50/72, 5/72, fig.dpi_scale_trans)
    ax = plt.subplot(grid2[0, 0])
    il = plot_label(ltr, il, ax, trans, fs_title)
    panel_varexp_future(ax, varexps_areas, tlags)

    ax = plt.subplot(grid2[0, 1])
    il = plot_label(ltr, il, ax, trans, fs_title)
    panel_future_decay(ax, varexps_areas, tlags)

    plt.show()
                
    if save_fig:
        fig.savefig(f'{data_path}figs/fig1_draft.pdf')


