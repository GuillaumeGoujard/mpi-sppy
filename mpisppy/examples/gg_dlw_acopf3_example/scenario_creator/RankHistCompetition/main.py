from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.RankHistObj import *
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.GMRF import to_minnesota_time
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.GMRF import GMRF
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.FilterClassNew import temp_median, no_filter
import mpisppy.examples.gg_dlw_acopf3_example.FilterClass as FilterClass
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.ARXandNormal import Normal, ARX

def main_function(start_time, end_time, number_of_days_fit, n_scenarios):
    dict_of_models = {
        "ARX": ARX(to_minnesota_time(datetime.datetime(2019, 11, 26, 0, 0)),
                   number_of_days_fit=number_of_days_fit, filter=FilterClass.temp_median()),
        "TGMRF": GMRF(number_of_days_fit=number_of_days_fit, filter=temp_median()),
        "RealFrontTGMRF": GMRF(number_of_days_fit=number_of_days_fit, filter=temp_median(), real_front=True),
        "Normal": Normal(number_of_days_fit=number_of_days_fit, filter=no_filter()),
        "SegmentedNormal": Normal(number_of_days_fit=number_of_days_fit, filter=temp_median()),
    }
    print("Models loaded are ", dict_of_models.keys())

    RH = RankHist(GMRF.tdata, dict_of_models, start_time, end_time, n_scenarios=n_scenarios, overfit=False)
    RH.compute_scenarios()
    RH.compute_daily_stats()
    RH.daily_horse_race(save=True)
    RH.fill_rank_histogram()
    RH.plot(savefig=True)


if __name__ == '__main__':
    number_of_days_fit = 10
    start_time = to_minnesota_time(datetime.datetime(2019, 12, 22, 0, 0))
    end_time = to_minnesota_time(datetime.datetime(2020, 1, 16, 0, 0))
    n_scenarios = 100
    main_function(start_time, end_time, number_of_days_fit, n_scenarios)
