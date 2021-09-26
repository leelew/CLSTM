def plot_time_series():

    plt.figure(figsize=(20, 5))
    ax = plt.subplot(111)

    plt.plot(y_pred_lstm, linewidth=3, color='blue', linestyle='-')
    plt.plot(y_pred_clstm_1, linewidth=3, color='green', linestyle='-')
    plt.plot(y_pred_rf, linewidth=3)
    #plt.plot(sm, linewidth=3, color = 'black', linestyle='--')
    plt.plot(y_test, linewidth=5, color='gray')

    plt.legend(['LSTM', 'Causal LSTM', 'RF', 'Observation'])
    ax.set_xlabel('Time (day)', fontsize=20)
    ax.set_ylabel('Soil moisture (Volumetric)', fontsize=20)

    ax1 = ax.twinx()
    test_df_ = test_df['P_F']*train_std['P_F']+train_mean['P_F']
    test_df_ = np.array(test_df_[:test_len])
    ax1.bar(x=np.arange(len(np.array(test_df_))),
            height=np.array(test_df_), color='red')
    ax1.set_ylim(0, max(test_df_)+50)

    ax1.set_xlabel('Time (day)', fontsize=20)
    ax1.set_ylabel('Precipitation (mm)', fontsize=20)

    plt.savefig(OUT_PATH + site_name + '/time_series_'+site_name+'.pdf')


def plot_scatter():
    def linear_(x, y):
        a, b = np.polyfit(x, y, deg=1)
        y_est = a * x + b
        y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))

        return y_est, y_err, a, b

    y_est_lstm, y_err_lstm, a_lstm, b_lstm = linear_(
        np.squeeze(y_pred_lstm), np.squeeze(y_test))

    y_est_clstm_1, y_err_clstm_1, a_clstm_1, b_clstm_1 = linear_(
        np.squeeze(y_pred_clstm_1), np.squeeze(y_test))
    y_est_clstm_2, y_err_clstm_2, a_clstm_2, b_clstm_2 = linear_(
        np.squeeze(y_pred_clstm_2), np.squeeze(y_test))
    y_est_clstm_3, y_err_clstm_3, a_clstm_3, b_clstm_3 = linear_(
        np.squeeze(y_pred_clstm_3), np.squeeze(y_test))

    y_est_rf, y_err_rf, a_rf, b_rf = linear_(
        np.squeeze(y_pred_rf), np.squeeze(y_test))
    #y_est_colm, y_err_colm, a_colm, b_colm = linear_(np.squeeze(sm), np.squeeze(y_test))

    min_, max_ = np.min(y_pred_lstm), np.max(y_pred_lstm)

    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(131)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)

    ax1.plot((0, 1), (0, 1), transform=ax1.transAxes,
             ls='--', c='k', label="1:1 line")
    ax1.scatter(y_pred_lstm, y_test)

    ax1.plot(y_pred_lstm, y_est_lstm, '-', color='red')
    ax1.fill_between(y_pred_lstm, y_est_lstm - y_err_lstm,
                     y_est_lstm + y_err_lstm, alpha=0.2)
    plt.xlim(min_-10, max_+15)
    plt.ylim(min_-10, max_+15)
    ax1.set_xlabel('LSTM', fontsize=20)
    ax1.set_ylabel('Fluxnet-'+site_name, fontsize=20)
    plt.text(min_-8, max_+8, '$R=%.2f$' %
             (pearsonr(np.squeeze(y_pred_lstm), np.squeeze(y_test))[0]))
    plt.text(min_-8, max_+6, '$RMSE=%.2f$' %
             (np.sqrt(mean_squared_error(np.squeeze(y_test), np.squeeze(y_pred_lstm)))))
    plt.text(min_-8, max_+4, '$MAE=%.2f$' %
             (mean_absolute_error(np.squeeze(y_test), np.squeeze(y_pred_lstm))))
    plt.text(min_-8, max_+2, '$Y = %.1fX + %.2f$' % (a_lstm, b_lstm))

    ax2 = plt.subplot(132)
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)

    ax2.scatter(y_pred_clstm_1, y_test)
    plt.xlim(min_-10, max_+15)
    plt.ylim(min_-10, max_+15)
    ax2.plot((0, 1), (0, 1), transform=ax2.transAxes,
             ls='--', c='k', label="1:1 line")

    ax2.plot(y_pred_clstm_1, y_est_clstm_1, '-', color='red')
    ax2.fill_between(y_pred_clstm_1, y_est_clstm_1 -
                     y_err_clstm_1, y_est_clstm_1 + y_err_clstm_1, alpha=0.2)
    ax2.set_xlabel('Causal LSTM', fontsize=20)
    plt.text(min_-8, max_+8, '$R=%.2f$' %
             (pearsonr(np.squeeze(y_pred_clstm_1), np.squeeze(y_test))[0]))
    plt.text(min_-8, max_+6, '$RMSE=%.2f$' %
             (np.sqrt(mean_squared_error(np.squeeze(y_test), np.squeeze(y_pred_clstm_1)))))
    plt.text(min_-8, max_+4, '$MAE=%.2f$' %
             (mean_absolute_error(np.squeeze(y_test), np.squeeze(y_pred_clstm_1))))
    plt.text(min_-8, max_+2, '$Y = %.1fX + %.2f$' % (a_clstm_1, b_clstm_1))

    """
        ax4 = plt.subplot(133)
        ax4.spines['top'].set_linewidth(2)
        ax4.spines['right'].set_linewidth(2)
        ax4.spines['left'].set_linewidth(2)
        ax4.spines['bottom'].set_linewidth(2)

        ax4.scatter(sm, y_test)
        plt.xlim(min_-10,max_+10)
        plt.ylim(min_-10,max_+10)
        ax4.plot((0, 1), (0, 1), transform=ax4.transAxes, ls='--',c='k', label="1:1 line")

        ax4.plot(sm, y_est_colm, '-', color='red')
        ax4.fill_between(sm, y_est_colm - y_err_colm, y_est_colm + y_err_colm, alpha=0.2)
        ax4.set_xlabel('CoLM',fontsize=20)

        plt.text(min_-8,max_+8, '$R=%.2f$' % (pearsonr(np.squeeze(sm), np.squeeze(y_test))[0]))
        plt.text(min_-8,max_+6, '$RMSE=%.2f$' % (np.sqrt(mean_squared_error(np.squeeze(sm), np.squeeze(y_test)))))
        plt.text(min_-8,max_+4, '$MAE=%.2f$' % (np.sqrt(mean_absolute_error(np.squeeze(sm), np.squeeze(y_test)))))
        plt.text(min_-8,max_+2, '$Y = %.1fX + %.2f$' % (a_colm, b_colm))
        """

    plt.savefig(OUT_PATH + site_name + '/scatter_'+site_name+'.pdf')
