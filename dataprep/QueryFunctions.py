def r2e_data_sapphire_presto(): 
    return """
with raw_user_list as ( 
    select 
        user_uuid as useruuid, 
        groups as cohort 
    from 
        marketing_ds_dev.brazil_uplift_purple_segment s 
    join 
        kirby_external_data.brazil_uplift_uuids_comms_sent_20190828 cam 
    on s.user_uuid = cam.uuid 
    where groups in ('control', 'treatment_a', 'treatment_b') 
), 

filter as (
    select 
        client_uuid, 
        count(*) as num_orders_90d 
    from dwh.fact_trip 
        where datestr between '{filter_start_date}' and '{filter_end_date}' 
        and is_completed = true 
    group by 1 
), 

ordered_filter as (
    select client_uuid 
    from filter 
    where num_orders_90d > 0 
), 

user_list as ( 
    select 
        r.useruuid, 
        case r.cohort when 'treatment_b' then 'treatment' when 'treatment_a' then 'treatment' else 'control' end as cohort 
    from 
        raw_user_list r 
    join 
        ordered_filter o 
    on o.client_uuid = r.useruuid 
), 
-- get orders from manually applied promos                                                                                                                                                          
manual_orders as (                                                                                                                                                                                  
    select                                                                                                                                                                                          
        u.useruuid, 
        u.cohort, 
        f.order_trip_uuid,                                                                                                                                                                          
        f.workflow_uuid,                                                                                                                                                                            
        f.original_eater_fare_usd as original_eater_fare_usd,                                                                                                                                       
        s.net_inflows_usd as net_inflows_usd                                                                                                                                                        
    from 
        user_list as u 
    left join 
        dwh.fact_eats_trip as f 
        on f.client_uuid = u.useruuid 
        and f.datestr >= '{start_date}' 
        and f.datestr <= '{end_date}' 
        and f.is_completed = true                                                                                                                                                                   
    left join                                                                                                                                                                                       
        secure_finance.fds_eats_workflow_metrics as s                                                                                                                                               
        on s.workflow_uuid = f.workflow_uuid                                                                                                                                                        
        and cast(s.datestr as date) >= (cast('{start_date}' as date) - interval '1' day)                                                                                                                 
        and cast(s.datestr as date) <= (cast('{end_date}' as date) + interval '1' day)                                                                                                                   
    ), 

-- compute stats for the manual apply group 
user_manual_stats as ( 
    select 
        u.useruuid, 
        max(u.cohort) as cohort, 
        count(distinct m.order_trip_uuid)   as manual_apply_orders, 
        sum(m.original_eater_fare_usd)      as manual_apply_gb, 
        sum(m.net_inflows_usd)              as manual_apply_ni 
    from 
        user_list as u 
    left join 
        manual_orders as m 
        on m.useruuid = u.useruuid 
    group by 
        1 
    ), 

rider_features as ( 
    select 
    rfl.uuid 
    -- GI rider feature library features
    , rfl.rating_2driver_min_avg_84d AS rating_2driver_min_avg_84d
    , rfl.trip_incomplete_total_84d AS trip_incomplete_total_84d
    , rfl.days_active_84d AS days_active_84d
    , rfl.days_since_trip_first_lifetime AS days_since_trip_first_lifetime
    , rfl.days_since_last_hard_churn_lifetime AS days_since_last_hard_churn_lifetime
    , rfl.days_since_last_soft_churn_lifetime AS days_since_last_soft_churn_lifetime
    , rfl.fare_max_sd_84d AS fare_max_sd_84d
    , rfl.churns_hard_lifetime AS churns_hard_lifetime
    , rfl.trips_lifetime AS trips_lifetime
    , rfl.fare_max_p50_84d AS fare_max_p50_84d
    , rfl.duration_session_pre_request_max_p50_84d AS duration_session_pre_request_max_p50_84d
    , rfl.trip_pool_per_x_84d AS trip_pool_per_x_84d
    , rfl.fare_total_win7d_sd_84d AS fare_total_win7d_sd_84d
    , rfl.trip_complete_win7d_sd_84d AS trip_complete_win7d_sd_84d
    , rfl.session_per_days_active_84d AS session_per_days_active_84d
    , rfl.churns_soft_lifetime AS churns_soft_lifetime
    , rfl.trip_complete_per_days_active_84d AS trip_complete_per_days_active_84d
    , rfl.trip_pool_prc_84d AS trip_pool_prc_84d
    , rfl.session_background_pre_request_prc_84d AS session_background_pre_request_prc_84d
    , rfl.session_lt_1m_prc_84d AS session_lt_1m_prc_84d
    , rfl.session_request_prc_84d AS session_request_prc_84d
    , rfl.duration_session_outside_total_prc_84d AS duration_session_outside_total_prc_84d
    , rfl.trip_x_prc_84d AS trip_x_prc_84d
    , rfl.days_since_trip_last_lifetime AS days_since_trip_last_lifetime
    
    -- categorical feature 
    --, UPPER(rfl.channel_signup_lifetime) AS channel_signup_lifetime 
    --, UPPER(rfl.device_os_primary_lifetime) AS device_os_primary_lifetime 
    --, CAST(rfl.promo_used_84d AS INT) AS promo_used_84d 
    , CAST(rfl.has_session_request_84d AS INT) AS has_session_request_84d
    , CAST(rfl.has_session_without_request_84d AS INT) AS has_session_without_request_84d
    from 
    gi_models.rider_flib rfl 
    where rfl.datestr = '{filter_end_date}' 
) 

select * from user_manual_stats s left join rider_features f on s.useruuid = f.uuid 
--select * from manual_stats where cohort = 'control' 
--select * from manual_stats where manual_apply_orders is not null and manual_apply_orders > 0 and cohort = 'control' 

--select * from decision_frame where groups = 'treatment_a' 
""".format(filter_start_date='2019-05-26', filter_end_date='2019-08-26', start_date='2019-08-28', end_date='2019-09-11') 

def rt_data_presto_first(): 
    return """
    with rider_trips as( 
        select 
            client_uuid, 
            count(*) as num_trips 
        from 
            ( 
                select 
                    client_uuid, 
                    uuid 
                from dwh.fact_trip 
                where 
                    is_completed = true 
                    and datestr between '2019-08-03' and '2019-11-03' 
            ) 
        group by client_uuid 
    ), 
    
    rider_filter as ( 
        select client_uuid 
        from 
            rider_trips 
        where 
            num_trips > 0 
    ), 
    
    all_rider_tags as ( 
        select rider_uuid, 
            tag_name, 
            city_id, 
            treatment 
        from (
                select
                  distinct uuid as rider_uuid
                  , tag_name
                  , cast(split(b.tag_name, '_') [5] as int) as city_id
                  , not (tag_name LIKE '%_Control') as treatment 
                from
                  kirby_external_data.rt_experiment_tags_20191024 b
                where
                  tag_name in ('RT_PROMO_EXP_2019-10-28_20_Control', 'RT_PROMO_EXP_2019-10-28_6_Control', 'RT_PROMO_EXP_2019-10-28_8_Control', 'RT_PROMO_EXP_2019-10-28_20_RT_100', 'RT_PROMO_EXP_2019-10-28_6_RT_100', 'RT_PROMO_EXP_2019-10-28_8_RT_100') 
            ) a 
            left join 
            rider_filter 
            on a.rider_uuid = rider_filter.client_uuid 
    ), 
    
    rt_rider_tags as (
        select
            rt.msg.rider_uuid AS rider_uuid
            , rt.msg.city_id AS city_id
            , rt.msg.event_type AS event_type
            , MAX(rt.msg.in_control) AS rider_in_control
            , MIN(rt.datestr) AS active_date
            , AVG(case when rt.msg.session_in_control then 1.0 else 0.0 end) AS pct_sessions_control
        from 
            rawdata.kafka_hp_personalized_rt_offer_engine_realtime_decision_nodedup AS rt
        where
            datestr BETWEEN '2019-11-04'
            and '2019-11-17'
            and rt.msg.city_id IN (8, 20, 6)
            and rt.msg.event_type = 1 -- only SELECT post trip trigger
            and NOT rt.msg.duplicated_promo
        group by
            1, 2, 3
    ), 
    
    rider_metrics as ( 
        select 
            rider_uuid as rider_uuid, 
            sum(coalesce(gross_bookings_usd,0)) as gross_bookings_usd, 
            sum(coalesce(net_billings_usd,0)) as net_billings_usd, 
            sum(coalesce(net_bookings_usd,0)) as net_bookings_usd, 
            sum(coalesce(spend_usd,0)) as spend_usd, 
            sum(coalesce(num_trips,0)) as num_trips, 
            sum(coalesce(variable_contribution_usd,0)) as variable_contribution_usd 
        from 
            personalization.riders_weekly_stats_core_metrics 
        where 
            datestr BETWEEN '2019-11-04' AND '2019-11-17' 
        group by 
            1
    ), 
    
    fsf_sessions as ( 
        select 
            MAX(city_id) as city_id, 
            session_id as session_id, 
            MAX(rider_id) as rider_id, 
            MAX(origin_hexagon_id_9) as begin_hex, 
            MAX(destination_hexagon_id_9) as dropoff_hex, 
            MAX(session_start_time_utc) as utc_timestamp, 
            MAX(timezone) as timezone, 
            MAX(ufp_distance_miles) as estimate_fare_distance_in_miles, 
            MAX(ufp_duration_minutes) as estimate_fare_duration_in_minutes, 
            MAX(origin_destination_haversine_mi) as origin_destination_haversine_miles, 
            MAX(origin_lat) as origin_lat, 
            MAX(origin_lng) as origin_lng, 
            MAX(destination_lat) as destination_lat, 
            MAX(destination_lng) as destination_lng, 
            MAX(datestr) as datestr, 
            MAX(case when vehicle_view_id in (232) and vvid_summary.session_last_requested = TRUE 
            and vvid_summary.last_job_uuid is not NULL 
            and session_summary.session_end_status = 'ON_TRIP' 
            then 1 else 0 end) as completed_trip_x 
        from 
            marketplace_fact.fact_session_fare 
        where 
            capacity = 1 
            and ufp_fare is not null 
    -- Test 
            and city_id in (6, 8, 20) 
            and datestr BETWEEN '2019-11-04' and '2019-11-17' 
        group by 
            session_id 
    ), 
    
    per_rider_sessions as ( 
        select 
            rider_id
            , city_id
            , count(distinct session_id) as num_sessions
        from 
            fsf_sessions --marketplace_fact.fact_session_fare
        --where
            --datestr BETWEEN '2019-11-04' AND '2019-11-17'
            --city_id in (8, 20, 6)
        group by 
            1, 2
    ), 
    
    rider_level_metrics as (
        select 
            all_rider_tags.rider_uuid, 
            tag_name,
            treatment, 
            all_rider_tags.city_id,
            pct_sessions_control,
            active_date,
            gross_bookings_usd,
            net_billings_usd,
            net_bookings_usd,
            spend_usd,
            num_trips,
            variable_contribution_usd,
            num_sessions
            -- GI rider feature library features
            , rfl.rating_2driver_min_avg_84d AS rating_2driver_min_avg_84d
            --, rfl.trip_incomplete_total_84d AS trip_incomplete_total_84d
            , rfl.days_active_84d AS days_active_84d
            , rfl.days_since_trip_first_lifetime AS days_since_trip_first_lifetime
            --, rfl.days_since_last_hard_churn_lifetime AS days_since_last_hard_churn_lifetime
            , rfl.days_since_last_soft_churn_lifetime AS days_since_last_soft_churn_lifetime
            --, rfl.fare_max_sd_84d AS fare_max_sd_84d
            , rfl.churns_hard_lifetime AS churns_hard_lifetime
            --, rfl.trips_lifetime AS trips_lifetime
            , rfl.fare_max_p50_84d AS fare_max_p50_84d
            --, rfl.duration_session_pre_request_max_p50_84d AS duration_session_pre_request_max_p50_84d
            --, rfl.trip_pool_per_x_84d AS trip_pool_per_x_84d
            , rfl.fare_total_win7d_sd_84d AS fare_total_win7d_sd_84d
            , rfl.trip_complete_win7d_sd_84d AS trip_complete_win7d_sd_84d
            --, rfl.session_per_days_active_84d AS session_per_days_active_84d
            --, rfl.churns_soft_lifetime AS churns_soft_lifetime
            --, rfl.trip_complete_per_days_active_84d AS trip_complete_per_days_active_84d
            , rfl.trip_pool_prc_84d AS trip_pool_prc_84d
            --, rfl.session_background_pre_request_prc_84d AS session_background_pre_request_prc_84d
            --, rfl.session_lt_1m_prc_84d AS session_lt_1m_prc_84d
            , rfl.session_request_prc_84d AS session_request_prc_84d
            --, rfl.duration_session_outside_total_prc_84d AS duration_session_outside_total_prc_84d
            --, rfl.trip_x_prc_84d AS trip_x_prc_84d
            , rfl.days_since_trip_last_lifetime AS days_since_trip_last_lifetime, 
            rfl.fare_promo_total_avg_84d, 
            rfl.fare_total_avg_84d, 
            rfl.surge_trip_avg_84d, 
            --rfl.fare_total_win7d_potential_84d, 
            rfl.fare_total_win28d_potential_84d, 
            --rfl.fare_lifetime, 
            --rfl.time_to_first_message_minutes_mean_lifetime, 
            rfl.ata_trip_max_avg_84d, 
            --rfl.eta_trip_max_avg_84d, 
            --rfl.trip_pool_matched_avg_84d, 
            --rfl.payment_cash_trip_total_84d, 
            rfl.duration_trip_total_p50_84d 
            -- categorical feature 
            --, UPPER(rfl.channel_signup_lifetime) AS channel_signup_lifetime 
            --, UPPER(rfl.device_os_primary_lifetime) AS device_os_primary_lifetime 
            , CAST(rfl.promo_used_84d AS INT) AS promo_used_84d 
            --, CAST(rfl.has_session_request_84d AS INT) AS has_session_request_84d 
            --, CAST(rfl.has_session_without_request_84d AS INT) AS has_session_without_request_84d 
        from 
            all_rider_tags 
            left join rider_metrics on all_rider_tags.rider_uuid = rider_metrics.rider_uuid 
            left join rt_rider_tags on all_rider_tags.rider_uuid = rt_rider_tags.rider_uuid 
            left join per_rider_sessions on per_rider_sessions.rider_id = all_rider_tags.rider_uuid and per_rider_sessions.city_id = all_rider_tags.city_id 
            left join gi_models.rider_flib as rfl on rfl.uuid = all_rider_tags.rider_uuid and rfl.datestr = '2019-11-03' 
    ), 
    
    joined_sessions as ( 
        select 
            f.*, 
            a.rider_uuid 
        from 
            fsf_sessions f 
            left join 
            (
                select rider_uuid, city_id 
                from 
                    all_rider_tags 
            ) a 
        on a.rider_uuid = f.rider_id and a.city_id = f.city_id 
    ), 
    
    filtered_sessions as (
        select * 
        from 
            joined_sessions 
        where rider_uuid is not null 
    ) 

    select 
        * 
    from 
        filtered_sessions f 
        left join 
        rider_level_metrics r 
    on r.rider_uuid = f.rider_id and f.city_id = r.city_id 
"""

def rxgy_data_sapphire_presto_featuremod2(label_dates_weekly, feature_date, city_ids): 
    return """
WITH label AS (
    SELECT 
        rider_uuid
        , proposal_uuid
        , city_id
        , proposal_start_datestr
        , CASE WHEN is_treatment = true THEN 'treatment' ELSE 'control' END AS cohort
        , CASE WHEN initial_tier = 0 THEN 0.0
            WHEN initial_tier = 1 THEN 100.0 * gy_low
            WHEN initial_tier = 2 THEN 100.0 * gy_high 
        END AS gy_initial
        , action_set_id
        , rx_low AS rx_low
        , rx_high AS rx_high
        , 100.0 * gy_low AS gy_low
        , 100.0 * gy_high AS gy_high
        , initial_tier
        , SUM(COALESCE(trip_count_7d, 0)) AS label_trip_28d
        , SUM(COALESCE(spend_usd_7d, 0.0)) AS label_cost_28d
        , SUM(COALESCE(gb_usd_7d, 0.0)) AS label_gb_28d 
        , SUM(COALESCE(vc_usd_7d, 0.0)) AS label_vc_28d
    FROM 
        personalization.rxgy_training_label
    
    WHERE 1=1 
        -- label_dates_weekly is the weekly Sunday datestr during 4w RxGy campaign, this is updated from update_query_params in vars.py
        AND datestr IN ({label_dates_weekly})  
        AND city_id IN ({city_ids})
        AND is_explore
        AND rx_high > 0
        AND (NOT is_boost_candidate OR is_boost_candidate IS NULL)
        AND (campaign_type IS NULL OR UPPER(campaign_type) NOT LIKE '%EAT%')
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
)

SELECT  
    --campaign info
    label.rider_uuid AS rider_uuid
    , label.city_id AS city_id
    , label.cohort AS cohort 
        
    -- GI rider feature library features
    , rfl.rating_2driver_min_avg_84d AS rating_2driver_min_avg_84d
    , rfl.trip_incomplete_total_84d AS trip_incomplete_total_84d
    , rfl.days_active_84d AS days_active_84d
    , rfl.days_since_trip_first_lifetime AS days_since_trip_first_lifetime
    , rfl.days_since_last_hard_churn_lifetime AS days_since_last_hard_churn_lifetime
    , rfl.days_since_last_soft_churn_lifetime AS days_since_last_soft_churn_lifetime
    , rfl.fare_max_sd_84d AS fare_max_sd_84d
    , rfl.churns_hard_lifetime AS churns_hard_lifetime
    , rfl.trips_lifetime AS trips_lifetime
    , rfl.fare_max_p50_84d AS fare_max_p50_84d
    , rfl.duration_session_pre_request_max_p50_84d AS duration_session_pre_request_max_p50_84d
    , rfl.trip_pool_per_x_84d AS trip_pool_per_x_84d
    , rfl.fare_total_win7d_sd_84d AS fare_total_win7d_sd_84d
    , rfl.trip_complete_win7d_sd_84d AS trip_complete_win7d_sd_84d
    , rfl.session_per_days_active_84d AS session_per_days_active_84d
    , rfl.churns_soft_lifetime AS churns_soft_lifetime
    , rfl.trip_complete_per_days_active_84d AS trip_complete_per_days_active_84d
    , rfl.trip_pool_prc_84d AS trip_pool_prc_84d
    , rfl.session_background_pre_request_prc_84d AS session_background_pre_request_prc_84d
    , rfl.session_lt_1m_prc_84d AS session_lt_1m_prc_84d
    , rfl.session_request_prc_84d AS session_request_prc_84d
    , rfl.duration_session_outside_total_prc_84d AS duration_session_outside_total_prc_84d
    , rfl.trip_x_prc_84d AS trip_x_prc_84d
    , rfl.days_since_trip_last_lifetime AS days_since_trip_last_lifetime, 
    rfl.fare_promo_total_avg_84d, 
    rfl.fare_total_avg_84d, 
    rfl.surge_trip_avg_84d, 
    rfl.fare_total_win7d_potential_84d, 
    rfl.fare_total_win28d_potential_84d, 
    rfl.fare_lifetime, 
    rfl.time_to_first_message_minutes_mean_lifetime, 
    rfl.ata_trip_max_avg_84d, 
    rfl.eta_trip_max_avg_84d, 
    rfl.trip_pool_matched_avg_84d, 
    rfl.payment_cash_trip_total_84d, 
    rfl.duration_trip_total_p50_84d
    
    -- categorical feature
    --, UPPER(rfl.channel_signup_lifetime) AS channel_signup_lifetime
    --, UPPER(rfl.device_os_primary_lifetime) AS device_os_primary_lifetime
    , CAST(rfl.promo_used_84d AS INT) AS promo_used_84d
    , CAST(rfl.has_session_request_84d AS INT) AS has_session_request_84d
    , CAST(rfl.has_session_without_request_84d AS INT) AS has_session_without_request_84d

    
    --promo feature
    , label.action_set_id AS action_set_id
    , label.rx_low AS rx_low
    , label.rx_high AS rx_high
    , label.gy_low AS gy_low
    , label.gy_high AS gy_high
    , label.gy_initial AS gy_initial
    , label.initial_tier AS initial_tier
    
    --label
    , label.label_trip_28d AS label_trip_28d
    , label.label_cost_28d AS label_cost_28d
    , label.label_gb_28d AS label_gb_28d 
    , label.label_vc_28d AS label_vc_28d 
FROM 
    label
JOIN 
    gi_models.rider_flib AS rfl  -- a daily pipeline with a lag of 3 days
ON 
    rfl.datestr = {feature_date}
    AND rfl.uuid = label.rider_uuid
""".format(label_dates_weekly=label_dates_weekly, feature_date=feature_date, city_ids=city_ids) 

def rxgy_data_sapphire_presto(label_dates_weekly, feature_date, city_ids):
    return """
WITH label AS (
    SELECT 
        rider_uuid
        , proposal_uuid
        , city_id
        , proposal_start_datestr
        , CASE WHEN is_treatment = true THEN 'treatment' ELSE 'control' END AS cohort
        , CASE WHEN initial_tier = 0 THEN 0.0
            WHEN initial_tier = 1 THEN 100.0 * gy_low
            WHEN initial_tier = 2 THEN 100.0 * gy_high 
        END AS gy_initial
        , action_set_id
        , rx_low AS rx_low
        , rx_high AS rx_high
        , 100.0 * gy_low AS gy_low
        , 100.0 * gy_high AS gy_high
        , initial_tier
        , SUM(COALESCE(trip_count_7d, 0)) AS label_trip_28d
        , SUM(COALESCE(spend_usd_7d, 0.0)) AS label_cost_28d
        , SUM(COALESCE(gb_usd_7d, 0.0)) AS label_gb_28d 
        , SUM(COALESCE(vc_usd_7d, 0.0)) AS label_vc_28d
    FROM 
        personalization.rxgy_training_label
    
    WHERE 1=1 
        -- label_dates_weekly is the weekly Sunday datestr during 4w RxGy campaign, this is updated from update_query_params in vars.py
        AND datestr IN ({label_dates_weekly})  
        AND city_id IN ({city_ids})
        AND is_explore
        AND rx_high > 0
        AND (NOT is_boost_candidate OR is_boost_candidate IS NULL)
        AND (campaign_type IS NULL OR UPPER(campaign_type) NOT LIKE '%EAT%')
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
)

SELECT  
    --campaign info
    label.rider_uuid AS rider_uuid
    , label.city_id AS city_id
    , label.cohort AS cohort
    
     -- GI rider feature library features
    
    -- GI rider feature library features
    , rfl.rating_2driver_min_avg_84d AS rating_2driver_min_avg_84d
    , rfl.trip_incomplete_total_84d AS trip_incomplete_total_84d
    , rfl.days_active_84d AS days_active_84d
    , rfl.days_since_trip_first_lifetime AS days_since_trip_first_lifetime
    , rfl.days_since_last_hard_churn_lifetime AS days_since_last_hard_churn_lifetime
    , rfl.days_since_last_soft_churn_lifetime AS days_since_last_soft_churn_lifetime
    , rfl.fare_max_sd_84d AS fare_max_sd_84d
    , rfl.churns_hard_lifetime AS churns_hard_lifetime
    , rfl.trips_lifetime AS trips_lifetime
    , rfl.fare_max_p50_84d AS fare_max_p50_84d
    , rfl.duration_session_pre_request_max_p50_84d AS duration_session_pre_request_max_p50_84d
    , rfl.trip_pool_per_x_84d AS trip_pool_per_x_84d
    , rfl.fare_total_win7d_sd_84d AS fare_total_win7d_sd_84d
    , rfl.trip_complete_win7d_sd_84d AS trip_complete_win7d_sd_84d
    , rfl.session_per_days_active_84d AS session_per_days_active_84d
    , rfl.churns_soft_lifetime AS churns_soft_lifetime
    , rfl.trip_complete_per_days_active_84d AS trip_complete_per_days_active_84d
    , rfl.trip_pool_prc_84d AS trip_pool_prc_84d
    , rfl.session_background_pre_request_prc_84d AS session_background_pre_request_prc_84d
    , rfl.session_lt_1m_prc_84d AS session_lt_1m_prc_84d
    , rfl.session_request_prc_84d AS session_request_prc_84d
    , rfl.duration_session_outside_total_prc_84d AS duration_session_outside_total_prc_84d
    , rfl.trip_x_prc_84d AS trip_x_prc_84d
    , rfl.days_since_trip_last_lifetime AS days_since_trip_last_lifetime
    
    -- categorical feature
    --, UPPER(rfl.channel_signup_lifetime) AS channel_signup_lifetime
    --, UPPER(rfl.device_os_primary_lifetime) AS device_os_primary_lifetime
    , CAST(rfl.promo_used_84d AS INT) AS promo_used_84d
    , CAST(rfl.has_session_request_84d AS INT) AS has_session_request_84d
    , CAST(rfl.has_session_without_request_84d AS INT) AS has_session_without_request_84d

    
    --promo feature
    , label.action_set_id AS action_set_id
    , label.rx_low AS rx_low
    , label.rx_high AS rx_high
    , label.gy_low AS gy_low
    , label.gy_high AS gy_high
    , label.gy_initial AS gy_initial
    , label.initial_tier AS initial_tier
    
    --label
    , label.label_trip_28d AS label_trip_28d
    , label.label_cost_28d AS label_cost_28d
    , label.label_gb_28d AS label_gb_28d 
    , label.label_vc_28d AS label_vc_28d 
FROM 
    label
JOIN 
    gi_models.rider_flib AS rfl  -- a daily pipeline with a lag of 3 days
ON 
    rfl.datestr = {feature_date}
    AND rfl.uuid = label.rider_uuid
""".format(label_dates_weekly=label_dates_weekly, feature_date=feature_date, city_ids=city_ids) 

def rxgy_data_sapphire_hive(columns_gi_rfl, label_dates_weekly, city_ids, feature_dates, days_shift): 
    # ETL: https://mlexplorer.uberinternal.com/project/Personalization%20Data%20ETL/workflow/4551936b-872c-42ad-b0f3-6b6273eef7e7
    return """
WITH label AS (
    SELECT 
        rider_uuid
        , proposal_uuid
        , city_id
        , proposal_start_datestr
        , CASE WHEN is_treatment = true THEN 'treatment' ELSE 'control' END AS cohort
        , CASE WHEN initial_tier = 0 THEN 0.0
            WHEN initial_tier = 1 THEN 100.0 * gy_low
            WHEN initial_tier = 2 THEN 100.0 * gy_high 
        END AS gy_initial
        , action_set_id
        , rx_low AS rx_low
        , rx_high AS rx_high
        , 100.0 * gy_low AS gy_low
        , 100.0 * gy_high AS gy_high
        , initial_tier
        , SUM(COALESCE(trip_count_7d, 0)) AS label_trip_28d
        , SUM(COALESCE(spend_usd_7d, 0.0)) AS label_cost_28d
    
    FROM 
        personalization.rxgy_training_label
    
    WHERE 1=1 
        -- label_dates_weekly is the weekly Sunday datestr during 4w RxGy campaign, this is updated from update_query_params in vars.py
        AND datestr IN ({label_dates_weekly})  
        AND city_id IN ({city_ids})
        AND is_explore
        AND rx_high > 0
        AND NOT is_boost_candidate
        AND (campaign_type IS NULL OR UPPER(campaign_type) NOT LIKE '%EAT%')
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
)

SELECT  
    --campaign info
    label.rider_uuid AS rider_uuid
    , label.city_id AS city_id
    , label.cohort AS cohort
    
     -- GI rider feature library features
    {columns_gi_rfl}
    
    --promo feature
    , label.action_set_id AS action_set_id
    , label.rx_low AS rx_low
    , label.rx_high AS rx_high
    , label.gy_low AS gy_low
    , label.gy_high AS gy_high
    , label.gy_initial AS gy_initial
    , label.initial_tier AS initial_tier
    
    --label
    , label.label_trip_28d AS label_trip_28d
    , label.label_cost_28d AS label_cost_28d
FROM 
    label 
JOIN 
    gi_models.rider_flib AS rfl  -- a daily pipeline with a lag of 3 days 
ON 
    rfl.datestr IN ({feature_dates})  
    AND rfl.uuid = label.rider_uuid
    -- feature datestr is 8d ahead of proposal start date (Monday), which is specified as label_dates in parameters
    -- ToDo: need to have a better handle
    AND rfl.datestr IN (
        DATE_ADD(label.proposal_start_datestr, {days_shift}), 
        DATE_ADD(label.proposal_start_datestr, {days_shift} + 1), -- for international case, the date will be one day
        DATE_ADD(label.proposal_start_datestr, {days_shift} + 2) -- for international case, the date will be one day
    ) 
""".format( 
    columns_gi_rfl=columns_gi_rfl, 
    label_dates_weekly=label_dates_weekly, 
    city_ids=city_ids, 
    feature_dates=feature_dates, 
    days_shift=days_shift 
) 

def rxgy_data_short(start_date, end_date, cities): 
    return """ 
with user_base as (
SELECT
    city_id
    , rider_uuid
    , proposal_uuid
    , proposal_start_datestr
    , cast((from_iso8601_date(proposal_start_datestr) - interval '8' day) as VARCHAR) as proposal_start_datestr_prev_sun
    , is_explore
    , is_treatment
    , rx_low
    , rx_high
    , gy_low
    , gy_high
    , initial_tier
    , SUM(trip_count_7d) AS trip_count_28d
    , SUM(net_billings_usd_7d) AS net_billings_usd_28d
    , SUM(gb_usd_7d) AS gb_usd_28d
    , SUM(spend_usd_7d) AS spend_usd_28d
    , SUM(vc_usd_7d) AS vc_usd_28d
    , COUNT(*) AS weeks_count
FROM
    personalization.rxgy_training_label
WHERE
    NOT is_boost_candidate
    AND datestr BETWEEN '{start_date}' AND '{end_date}'
    AND city_id IN ({city_ids}) 
    AND is_explore = True
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12 
), 

features as ( 
    select 
        uuid, 
        churns_hard_lifetime, 
        days_active_lifetime, 
        days_since_trip_first_lifetime, 
        fare_lifetime, 
        days_active_84d, 
        trip_pool_matched_avg_84d, 
        fare_promo_total_avg_84d, 
        fare_total_avg_84d, 
        ata_trip_max_avg_84d, 
        eta_trip_max_avg_84d, 
        rating_2rider_total_avg_84d, 
        surge_trip_avg_84d, 
        fare_total_win7d_potential_84d, 
        trip_complete_win7d_potential_84d, 
        trip_total_win7d_potential_84d, 
        fare_total_win28d_potential_84d, 
        trip_complete_win28d_potential_84d, 
        trip_total_win28d_potential_84d, 
        datestr 
    from 
        gi_models.rider_flib 
    where 
        datestr between '{start_date}' and '{end_date}' 
        and city_id in ({city_ids}) 
) 

select * from 
user_base u 
left join 
features f  
on u.rider_uuid = f.uuid 
--and u.proposal_start_datestr_prev_sun = f.datestr 

""".format(start_date=start_date, end_date=end_date, city_ids=cities) 

def rxgy_data_short_groupby(start_date, end_date, cities): 
    return """ 
with user_base as (
SELECT
    city_id
    , rider_uuid
    , proposal_uuid
    , proposal_start_datestr
    , cast((from_iso8601_date(proposal_start_datestr) - interval '8' day) as VARCHAR) as proposal_start_datestr_prev_sun
    , is_explore
    , is_treatment
    , rx_low
    , rx_high
    , gy_low
    , gy_high
    , initial_tier
    , SUM(trip_count_7d) AS trip_count_28d
    , SUM(net_billings_usd_7d) AS net_billings_usd_28d
    , SUM(gb_usd_7d) AS gb_usd_28d
    , SUM(spend_usd_7d) AS spend_usd_28d
    , SUM(vc_usd_7d) AS vc_usd_28d
    , COUNT(*) AS weeks_count
FROM
    personalization.rxgy_training_label
WHERE
    NOT is_boost_candidate
    AND datestr BETWEEN '{start_date}' AND '{end_date}'
    AND city_id IN ({city_ids}) 
    AND is_explore = True
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12 
), 

features as ( 
    select 
        uuid, 
        churns_hard_lifetime, 
        days_active_lifetime, 
        days_since_trip_first_lifetime, 
        fare_lifetime, 
        days_active_84d, 
        trip_pool_matched_avg_84d, 
        fare_promo_total_avg_84d, 
        fare_total_avg_84d, 
        ata_trip_max_avg_84d, 
        eta_trip_max_avg_84d, 
        rating_2rider_total_avg_84d, 
        surge_trip_avg_84d, 
        fare_total_win7d_potential_84d, 
        trip_complete_win7d_potential_84d, 
        trip_total_win7d_potential_84d, 
        fare_total_win28d_potential_84d, 
        trip_complete_win28d_potential_84d, 
        trip_total_win28d_potential_84d, 
        datestr 
    from 
        gi_models.rider_flib 
    where 
        datestr between '{start_date}' and '{end_date}' 
        and city_id in ({city_ids}) 
) 

, near_final as ( 
select * from 
user_base u 
left join 
features f  
on u.rider_uuid = f.uuid 
--and u.proposal_start_datestr_prev_sun = f.datestr 
) 

select 
        uuid, 
        proposal_uuid 
        , proposal_start_datestr 
        , cast((from_iso8601_date(proposal_start_datestr) - interval '8' day) as VARCHAR) as proposal_start_datestr_prev_sun
        , is_explore
        , is_treatment
        , rx_low
        , rx_high
        , gy_low
        , gy_high
        , initial_tier
        , trip_count_28d
        , net_billings_usd_28d 
        , gb_usd_28d 
        , spend_usd_28d 
        , vc_usd_28d 
        , weeks_count 
        , churns_hard_lifetime, 
        avg(days_active_lifetime) as days_active_lifetime, 
        avg(days_since_trip_first_lifetime) as days_since_trip_first_lifetime, 
        avg(fare_lifetime) as fare_lifetime, 
        avg(days_active_84d) as days_active_84d, 
        avg(trip_pool_matched_avg_84d) as trip_pool_matched_avg_84d, 
        avg(fare_promo_total_avg_84d) as fare_promo_total_avg_84d, 
        avg(fare_total_avg_84d) as fare_total_avg_84d, 
        avg(ata_trip_max_avg_84d) as ata_trip_max_avg_84d, 
        avg(eta_trip_max_avg_84d) as eta_trip_max_avg_84d, 
        avg(rating_2rider_total_avg_84d) as rating_2rider_total_avg_84d, 
        avg(surge_trip_avg_84d) as surge_trip_avg_84d, 
        avg(fare_total_win7d_potential_84d) as fare_total_win7d_potential_84d, 
        avg(trip_complete_win7d_potential_84d) as trip_complete_win7d_potential_84d, 
        avg(trip_total_win7d_potential_84d) as trip_total_win7d_potential_84d, 
        avg(fare_total_win28d_potential_84d) as fare_total_win28d_potential_84d, 
        avg(trip_complete_win28d_potential_84d) as trip_complete_win28d_potential_84d, 
        avg(trip_total_win28d_potential_84d) as trip_total_win28d_potential_84d, 
        max(datestr) as datestr 
from near_final 
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 
""".format(start_date=start_date, end_date=end_date, city_ids=cities) 

def subs_upsell_data_short(yesterday_ds, cities): 
    return """
with grouped as (
    select 
        upsell_user_id, 
        max(city_id) as city_id, 
        max(upsell_label_15d) as buy_pass_15d, 
        sum(cast(in_session_trip_dropoff_label as double)) as total_trip_dropoff, 
        sum(cast(current_request_dropoff_label as double)) as total_request_dropoff, 
        max(upsell_session_id) as upsell_session_id,
        max(origin_hexagon) as origin_hexagon,
        max(destination_hexagon) as destination_hexagon,
        max(upsell_label_15d) as upsell_label_15d,
        max(landing_page_label) as landing_page_label,
        max(get_pass_label) as get_pass_label,
        max(pass_purchase_label) as pass_purchase_label,
        max(current_request_dropoff_label) as current_request_dropoff_label,
        max(in_session_trip_dropoff_label) as in_session_trip_dropoff_label,
        max(trip_complete_84d) as trip_complete_84d,
        max(trip_complete_per_days_active_84d) as trip_complete_per_days_active_84d,
        max(promo_used_84d) as promo_used_84d,
        max(trip_x_prc_84d) as trip_x_prc_84d,
        max(trip_pool_prc_84d) as trip_pool_prc_84d,
        max(trip_pool_per_x_84d) as trip_pool_per_x_84d,
        max(session_per_days_active_84d) as session_per_days_active_84d,
        max(session_request_prc_84d) as session_request_prc_84d,
        max(session_background_pre_request_prc_84d) as session_background_pre_request_prc_84d,
        max(has_session_request_84d) as has_session_request_84d,
        max(duration_session_outside_total_prc_84d) as duration_session_outside_total_prc_84d,
        max(has_session_without_request_84d) as has_session_without_request_84d,
        max(payment_cash_trip_prc_84d) as payment_cash_trip_prc_84d,
        max(surge_trip_prc_84d) as surge_trip_prc_84d,
        max(ufp_trip_not_honored_prc_84d) as ufp_trip_not_honored_prc_84d,
        max(ufp_trip_total_prc_84d) as ufp_trip_total_prc_84d,
        max(trip_promo_prc_84d) as trip_promo_prc_84d,
        max(trip_complete_prc_84d) as trip_complete_prc_84d,
        max(trip_rider_cancelled_prc_84d) as trip_rider_cancelled_prc_84d,
        max(trip_driver_cancelled_prc_84d) as trip_driver_cancelled_prc_84d,
        max(request_to_trip_prc_84d) as request_to_trip_prc_84d,
        max(days_session_request_prc_84d) as days_session_request_prc_84d,
        max(trips_lifetime) as trips_lifetime,
        max(trip_complete_win7d_potential_84d) as trip_complete_win7d_potential_84d,
        max(days_since_trip_first_lifetime) as days_since_trip_first_lifetime,
        max(trip_complete_win28d_potential_84d) as trip_complete_win28d_potential_84d,
        max(fare_total_win7d_potential_84d) as fare_total_win7d_potential_84d,
        max(trip_total_total_84d) as trip_total_total_84d,
        max(fare_total_win28d_potential_84d) as fare_total_win28d_potential_84d,
        max(days_since_last_soft_churn_lifetime) as days_since_last_soft_churn_lifetime,
        max(days_active_84d) as days_active_84d,
        max(days_since_last_hard_churn_lifetime) as days_since_last_hard_churn_lifetime,
        max(session_lt_1m_prc_84d) as session_lt_1m_prc_84d,
        max(fare_max_p50_84d) as fare_max_p50_84d,
        max(uber_preferred_score) as uber_preferred_score,
        max(datestr) as datestr
    from 
        subscriptions.upsell_targeting_features_label_training 
    group by upsell_user_id 
), 

labels_with_features as (
select 
    case
        when (
            (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(upsell_user_id || chr(1079) || 'super_layer_subs_holdout')))), 1,  8 ), 16) % 100) * (96*96*96 % 100) + 
            (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(upsell_user_id || chr(1079) || 'super_layer_subs_holdout')))), 9,  8 ), 16) % 100) * (96*96 % 100) + 
            (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(upsell_user_id || chr(1079) || 'super_layer_subs_holdout')))), 17, 8 ), 16) % 100) * (96 % 100) + 
            (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(upsell_user_id || chr(1079) || 'super_layer_subs_holdout')))), 25, 8 ), 16) % 100)
            ) % 100 < 9.5 then 'control'
        else 'treatment'
    end as cohort, -- new cohort 
    upsell_user_id, 
    buy_pass_15d, 
    total_trip_dropoff, 
    total_request_dropoff, 
    upsell_session_id,
    origin_hexagon,
    destination_hexagon,
    landing_page_label,
    get_pass_label,
    pass_purchase_label,
    trip_complete_84d,
    trip_complete_per_days_active_84d,
    promo_used_84d,
    trip_x_prc_84d,
    trip_pool_prc_84d,
    trip_pool_per_x_84d,
    session_per_days_active_84d,
    session_request_prc_84d,
    session_background_pre_request_prc_84d,
    has_session_request_84d,
    duration_session_outside_total_prc_84d,
    has_session_without_request_84d,
    payment_cash_trip_prc_84d,
    surge_trip_prc_84d,
    ufp_trip_not_honored_prc_84d,
    ufp_trip_total_prc_84d,
    trip_promo_prc_84d,
    trip_complete_prc_84d,
    trip_rider_cancelled_prc_84d,
    trip_driver_cancelled_prc_84d,
    request_to_trip_prc_84d,
    days_session_request_prc_84d,
    trips_lifetime,
    trip_complete_win7d_potential_84d,
    days_since_trip_first_lifetime,
    trip_complete_win28d_potential_84d,
    fare_total_win7d_potential_84d,
    trip_total_total_84d,
    fare_total_win28d_potential_84d,
    days_since_last_soft_churn_lifetime,
    days_active_84d,
    days_since_last_hard_churn_lifetime,
    session_lt_1m_prc_84d,
    fare_max_p50_84d,
    uber_preferred_score,
    datestr
from 
    grouped 
where city_id in ({cities}) 
), 

all_subs_records as ( 
    SELECT 
        datestr, 
        session_id as session_id, 
        rider_id as bought_user_uuid, 
        city_id, 
        min(ts) as ts 
    FROM subscriptions.uberplus_event_user 
    WHERE 
        datestr between cast((date '{yesterday_ds}' - interval '15' day) as VARCHAR) and cast((date '{yesterday_ds}') as VARCHAR) 
        --date_add('{yesterday_ds}' , -15) and '{yesterday_ds}' 
        AND name in ('uber_pass_tap_purchase', 'subs_purchase_screen_purchase') 
        AND city_id IN (SELECT DISTINCT city_id FROM subscriptions.upsell_city_offer_info) --({cities}) --
    GROUP BY 1, 2, 3, 4 
) 

, near_final as (
select 
    case when (a.bought_user_uuid is null) then 0.0 else 1.0 end as buy_pass_15d_all, 
    random() as rnd_num, 
    l.* 
from 
    labels_with_features l 
left join 
    all_subs_records a 
    on 
    l.upsell_user_id = a.bought_user_uuid 
) 

, nf1 as ( 
select * 
from near_final 
where cohort = 'treatment' 
and rnd_num < 0.1 
) 

, nf2 as (
select * 
from near_final
where cohort = 'control' 
) 

, unioned as ( 
select * from 
nf1 union all select * from nf2 
) 

select * from unioned 
""".format(yesterday_ds=yesterday_ds, cities=cities) 


def subs_upsell_data(yesterday_ds, cities): 
    return """
with grouped as (
    select 
        upsell_user_id, 
        max(city_id) as city_id, 
        max(upsell_label_15d) as buy_pass_15d, 
        sum(cast(in_session_trip_dropoff_label as double)) as total_trip_dropoff, 
        sum(cast(current_request_dropoff_label as double)) as total_request_dropoff, 
        max(upsell_session_id) as upsell_session_id,
        max(origin_hexagon) as origin_hexagon,
        max(destination_hexagon) as destination_hexagon,
        max(upsell_label_15d) as upsell_label_15d,
        max(landing_page_label) as landing_page_label,
        max(get_pass_label) as get_pass_label,
        max(pass_purchase_label) as pass_purchase_label,
        max(current_request_dropoff_label) as current_request_dropoff_label,
        max(in_session_trip_dropoff_label) as in_session_trip_dropoff_label,
        max(trip_complete_84d) as trip_complete_84d,
        max(trip_complete_per_days_active_84d) as trip_complete_per_days_active_84d,
        max(promo_used_84d) as promo_used_84d,
        max(trip_x_prc_84d) as trip_x_prc_84d,
        max(trip_pool_prc_84d) as trip_pool_prc_84d,
        max(trip_pool_per_x_84d) as trip_pool_per_x_84d,
        max(session_per_days_active_84d) as session_per_days_active_84d,
        max(session_request_prc_84d) as session_request_prc_84d,
        max(session_background_pre_request_prc_84d) as session_background_pre_request_prc_84d,
        max(has_session_request_84d) as has_session_request_84d,
        max(duration_session_outside_total_prc_84d) as duration_session_outside_total_prc_84d,
        max(has_session_without_request_84d) as has_session_without_request_84d,
        max(payment_cash_trip_prc_84d) as payment_cash_trip_prc_84d,
        max(surge_trip_prc_84d) as surge_trip_prc_84d,
        max(ufp_trip_not_honored_prc_84d) as ufp_trip_not_honored_prc_84d,
        max(ufp_trip_total_prc_84d) as ufp_trip_total_prc_84d,
        max(trip_promo_prc_84d) as trip_promo_prc_84d,
        max(trip_complete_prc_84d) as trip_complete_prc_84d,
        max(trip_rider_cancelled_prc_84d) as trip_rider_cancelled_prc_84d,
        max(trip_driver_cancelled_prc_84d) as trip_driver_cancelled_prc_84d,
        max(request_to_trip_prc_84d) as request_to_trip_prc_84d,
        max(days_session_request_prc_84d) as days_session_request_prc_84d,
        max(trips_lifetime) as trips_lifetime,
        max(trip_complete_win7d_potential_84d) as trip_complete_win7d_potential_84d,
        max(days_since_trip_first_lifetime) as days_since_trip_first_lifetime,
        max(trip_complete_win28d_potential_84d) as trip_complete_win28d_potential_84d,
        max(fare_total_win7d_potential_84d) as fare_total_win7d_potential_84d,
        max(trip_total_total_84d) as trip_total_total_84d,
        max(fare_total_win28d_potential_84d) as fare_total_win28d_potential_84d,
        max(days_since_last_soft_churn_lifetime) as days_since_last_soft_churn_lifetime,
        max(days_active_84d) as days_active_84d,
        max(days_since_last_hard_churn_lifetime) as days_since_last_hard_churn_lifetime,
        max(session_lt_1m_prc_84d) as session_lt_1m_prc_84d,
        max(fare_max_p50_84d) as fare_max_p50_84d,
        max(uber_preferred_score) as uber_preferred_score,
        max(home_city) as home_city,
        max(pct_sessions_peak_hour) as pct_sessions_peak_hour,
        max(pct_sessions_weekend) as pct_sessions_weekend,
        max(pct_sessions_surged) as pct_sessions_surged,
        max(avg_surge_multiplier) as avg_surge_multiplier,
        max(p10_surge_multiplier) as p10_surge_multiplier,
        max(p50_surge_multiplier) as p50_surge_multiplier,
        max(p90_surge_multiplier) as p90_surge_multiplier,
        max(tot_potential_saving_subs) as tot_potential_saving_subs,
        max(avg_potential_saving_subs) as avg_potential_saving_subs,
        max(p10_potential_saving_subs) as p10_potential_saving_subs,
        max(p50_potential_saving_subs) as p50_potential_saving_subs,
        max(p90_potential_saving_subs) as p90_potential_saving_subs,
        max(avg_session_eta_last) as avg_session_eta_last,
        max(p10_session_eta_last) as p10_session_eta_last,
        max(p50_session_eta_last) as p50_session_eta_last,
        max(p90_session_eta_last) as p90_session_eta_last,
        max(prob_has_promotion) as prob_has_promotion,
        max(prob_has_subscription) as prob_has_subscription,
        max(avg_promotions) as avg_promotions,
        max(avg_subscriptions) as avg_subscriptions,
        max(num_subs) as num_subs,
        max(num_refunds) as num_refunds,
        max(time_since_last_subs) as time_since_last_subs,
        max(time_since_refund) as time_since_refund,
        max(accum_upsell_impression) as accum_upsell_impression,
        max(accum_landing_page_view) as accum_landing_page_view,
        max(accum_landing_page_get_pass) as accum_landing_page_get_pass,
        max(pcor_activated) as pcor_activated,
        max(pcor_eligible) as pcor_eligible,
        max(week_of_year) as week_of_year,
        max(weekday_local) as weekday_local,
        max(hour_of_day_local) as hour_of_day_local,
        max(hour_of_week_local) as hour_of_week_local,
        max(path_type) as path_type,
        max(is_traveler) as is_traveler,
        max(is_eater) as is_eater,
        max(x_ufp_fare_before_sub_promotion) as x_ufp_fare_before_sub_promotion,
        max(x_eta) as x_eta,
        max(x_surge) as x_surge,
        max(x_estimate_fare_distance_in_miles) as x_estimate_fare_distance_in_miles,
        max(x_estimate_fare_duration_in_minutes) as x_estimate_fare_duration_in_minutes,
        max(x_promotion) as x_promotion,
        max(x_rsp_multiplier) as x_rsp_multiplier,
        max(x_rsp_amount) as x_rsp_amount,
        max(x_tolls_surcharges) as x_tolls_surcharges,
        max(x_predicted_fare_rd) as x_predicted_fare_rd,
        max(x_predicted_saving) as x_predicted_saving,
        max(sr_ufp_fare_before_sub_promotion) as sr_ufp_fare_before_sub_promotion,
        max(sr_surge) as sr_surge,
        max(sr_estimate_fare_distance_in_miles) as sr_estimate_fare_distance_in_miles,
        max(sr_estimate_fare_duration_in_minutes) as sr_estimate_fare_duration_in_minutes,
        max(sr_promotion) as sr_promotion,
        max(sr_rsp_multiplier) as sr_rsp_multiplier,
        max(sr_rsp_amount) as sr_rsp_amount,
        max(sr_tolls_surcharges) as sr_tolls_surcharges,
        max(sr_predicted_fare_rd) as sr_predicted_fare_rd,
        max(sr_predicted_saving) as sr_predicted_saving,
        max(total_ufp_by_path) as total_ufp_by_path,
        max(total_ufp_by_origin) as total_ufp_by_origin,
        max(total_ufp_by_destination) as total_ufp_by_destination,
        max(pct_ufp_by_path) as pct_ufp_by_path,
        max(pct_ufp_by_origin) as pct_ufp_by_origin,
        max(pct_ufp_by_destination) as pct_ufp_by_destination,
        max(avg_surged_by_origin) as avg_surged_by_origin,
        max(num_completed_trips) as num_completed_trips,
        max(num_non_completed_trips) as num_non_completed_trips,
        max(total_trips) as total_trips,
        max(average_completed_fare) as average_completed_fare,
        max(average_non_completed_fare) as average_non_completed_fare,
        max(average_completed_base_fare) as average_completed_base_fare,
        max(average_non_completed_base_fare) as average_non_completed_base_fare,
        max(average_completed_trip_distance_miles) as average_completed_trip_distance_miles,
        max(average_completed_fare_distance_miles) as average_completed_fare_distance_miles,
        max(average_completed_fare_duration_minutes) as average_completed_fare_duration_minutes,
        max(c2r) as c2r,
        max(num_unique_riders) as num_unique_riders,
        max(avg_surge_multiplier_on_completed_trips) as avg_surge_multiplier_on_completed_trips,
        max(avg_surge_multiplier_on_non_completed_trips) as avg_surge_multiplier_on_non_completed_trips,
        max(pct_surged_complete_trips) as pct_surged_complete_trips,
        max(pct_surged_non_complete_trips) as pct_surged_non_complete_trips,
        max(avg_completed_eta) as avg_completed_eta,
        max(avg_non_completed_eta) as avg_non_completed_eta,
        max(pct_trips_on_time) as pct_trips_on_time,
        max(pct_completed_trips_with_promotion) as pct_completed_trips_with_promotion,
        max(pct_non_completed_trips_with_promotion) as pct_non_completed_trips_with_promotion,
        max(pct_completed_trips_commute_hours) as pct_completed_trips_commute_hours,
        max(pct_non_completed_trips_commute_hours) as pct_non_completed_trips_commute_hours,
        max(pct_completed_trips_weekend) as pct_completed_trips_weekend,
        max(pct_non_completed_trips_weekend) as pct_non_completed_trips_weekend,
        max(avg_completed_trip_to_sphere_distance_ratio) as avg_completed_trip_to_sphere_distance_ratio,
        max(avg_non_completed_trip_to_sphere_distance_ratio) as avg_non_completed_trip_to_sphere_distance_ratio,
        max(pct_airport_pickup_completed_trips) as pct_airport_pickup_completed_trips,
        max(pct_airport_pickup_non_completed_trips) as pct_airport_pickup_non_completed_trips,
        max(pct_airport_dropoff_completed_trips) as pct_airport_dropoff_completed_trips,
        max(pct_airport_dropoff_non_completed_trips) as pct_airport_dropoff_non_completed_trips,
        max(pct_sub_completed_trips) as pct_sub_completed_trips,
        max(pct_sub_non_completed_trips) as pct_sub_non_completed_trips,
        max(pickup_trip_start_num_completed_trips) as pickup_trip_start_num_completed_trips,
        max(pickup_trip_start_c2r) as pickup_trip_start_c2r,
        max(pickup_trip_start_pct_x_trips) as pickup_trip_start_pct_x_trips,
        max(pickup_trip_start_num_unique_riders) as pickup_trip_start_num_unique_riders,
        max(pickup_trip_start_avg_surge_multiplier_on_requests) as pickup_trip_start_avg_surge_multiplier_on_requests,
        max(pickup_trip_start_pct_surged_trips) as pickup_trip_start_pct_surged_trips,
        max(pickup_trip_start_avg_eta) as pickup_trip_start_avg_eta,
        max(pickup_trip_start_tot_ufp) as pickup_trip_start_tot_ufp,
        max(pickup_trip_start_avg_trip_distance_miles) as pickup_trip_start_avg_trip_distance_miles,
        max(pickup_trip_start_pct_trips_on_time) as pickup_trip_start_pct_trips_on_time,
        max(pickup_trip_start_pct_trips_with_promotion) as pickup_trip_start_pct_trips_with_promotion,
        max(pickup_trip_start_pct_trips_commute_hours) as pickup_trip_start_pct_trips_commute_hours,
        max(pickup_trip_start_pct_trips_weekend) as pickup_trip_start_pct_trips_weekend,
        max(pickup_trip_start_avg_trip_to_sphere_distance_ratio) as pickup_trip_start_avg_trip_to_sphere_distance_ratio,
        max(pickup_trip_start_pct_airport_trips) as pickup_trip_start_pct_airport_trips,
        max(pickup_trip_end_num_completed_trips) as pickup_trip_end_num_completed_trips,
        max(pickup_trip_end_c2r) as pickup_trip_end_c2r,
        max(pickup_trip_end_pct_x_trips) as pickup_trip_end_pct_x_trips,
        max(pickup_trip_end_num_unique_riders) as pickup_trip_end_num_unique_riders,
        max(pickup_trip_end_avg_surge_multiplier_on_requests) as pickup_trip_end_avg_surge_multiplier_on_requests,
        max(pickup_trip_end_pct_surged_trips) as pickup_trip_end_pct_surged_trips,
        max(pickup_trip_end_avg_eta) as pickup_trip_end_avg_eta,
        max(pickup_trip_end_tot_ufp) as pickup_trip_end_tot_ufp,
        max(pickup_trip_end_avg_trip_distance_miles) as pickup_trip_end_avg_trip_distance_miles,
        max(pickup_trip_end_pct_trips_on_time) as pickup_trip_end_pct_trips_on_time,
        max(pickup_trip_end_pct_trips_with_promotion) as pickup_trip_end_pct_trips_with_promotion,
        max(pickup_trip_end_pct_trips_commute_hours) as pickup_trip_end_pct_trips_commute_hours,
        max(pickup_trip_end_pct_trips_weekend) as pickup_trip_end_pct_trips_weekend,
        max(pickup_trip_end_avg_trip_to_sphere_distance_ratio) as pickup_trip_end_avg_trip_to_sphere_distance_ratio,
        max(pickup_trip_end_pct_airport_trips) as pickup_trip_end_pct_airport_trips,
        max(dropoff_trip_start_num_completed_trips) as dropoff_trip_start_num_completed_trips,
        max(dropoff_trip_start_c2r) as dropoff_trip_start_c2r,
        max(dropoff_trip_start_pct_x_trips) as dropoff_trip_start_pct_x_trips,
        max(dropoff_trip_start_num_unique_riders) as dropoff_trip_start_num_unique_riders,
        max(dropoff_trip_start_avg_surge_multiplier_on_requests) as dropoff_trip_start_avg_surge_multiplier_on_requests,
        max(dropoff_trip_start_pct_surged_trips) as dropoff_trip_start_pct_surged_trips,
        max(dropoff_trip_start_avg_eta) as dropoff_trip_start_avg_eta,
        max(dropoff_trip_start_tot_ufp) as dropoff_trip_start_tot_ufp,
        max(dropoff_trip_start_avg_trip_distance_miles) as dropoff_trip_start_avg_trip_distance_miles,
        max(dropoff_trip_start_pct_trips_on_time) as dropoff_trip_start_pct_trips_on_time,
        max(dropoff_trip_start_pct_trips_with_promotion) as dropoff_trip_start_pct_trips_with_promotion,
        max(dropoff_trip_start_pct_trips_commute_hours) as dropoff_trip_start_pct_trips_commute_hours,
        max(dropoff_trip_start_pct_trips_weekend) as dropoff_trip_start_pct_trips_weekend,
        max(dropoff_trip_start_avg_trip_to_sphere_distance_ratio) as dropoff_trip_start_avg_trip_to_sphere_distance_ratio,
        max(dropoff_trip_start_pct_airport_trips) as dropoff_trip_start_pct_airport_trips,
        max(dropoff_trip_end_num_completed_trips) as dropoff_trip_end_num_completed_trips,
        max(dropoff_trip_end_c2r) as dropoff_trip_end_c2r,
        max(dropoff_trip_end_pct_x_trips) as dropoff_trip_end_pct_x_trips,
        max(dropoff_trip_end_num_unique_riders) as dropoff_trip_end_num_unique_riders,
        max(dropoff_trip_end_avg_surge_multiplier_on_requests) as dropoff_trip_end_avg_surge_multiplier_on_requests,
        max(dropoff_trip_end_pct_surged_trips) as dropoff_trip_end_pct_surged_trips,
        max(dropoff_trip_end_avg_eta) as dropoff_trip_end_avg_eta,
        max(dropoff_trip_end_tot_ufp) as dropoff_trip_end_tot_ufp,
        max(dropoff_trip_end_avg_trip_distance_miles) as dropoff_trip_end_avg_trip_distance_miles,
        max(dropoff_trip_end_pct_trips_on_time) as dropoff_trip_end_pct_trips_on_time,
        max(dropoff_trip_end_pct_trips_with_promotion) as dropoff_trip_end_pct_trips_with_promotion,
        max(dropoff_trip_end_pct_trips_commute_hours) as dropoff_trip_end_pct_trips_commute_hours,
        max(dropoff_trip_end_pct_trips_weekend) as dropoff_trip_end_pct_trips_weekend,
        max(dropoff_trip_end_avg_trip_to_sphere_distance_ratio) as dropoff_trip_end_avg_trip_to_sphere_distance_ratio,
        max(dropoff_trip_end_pct_airport_trips) as dropoff_trip_end_pct_airport_trips,
        max(datestr) as datestr
    from 
        subscriptions.upsell_targeting_features_label_training 
    group by upsell_user_id 
), 

labels_with_features as (
select 
    case
        when (
            (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(upsell_user_id || chr(1079) || 'super_layer_subs_holdout')))), 1,  8 ), 16) % 100) * (96*96*96 % 100) + 
            (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(upsell_user_id || chr(1079) || 'super_layer_subs_holdout')))), 9,  8 ), 16) % 100) * (96*96 % 100) + 
            (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(upsell_user_id || chr(1079) || 'super_layer_subs_holdout')))), 17, 8 ), 16) % 100) * (96 % 100) + 
            (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(upsell_user_id || chr(1079) || 'super_layer_subs_holdout')))), 25, 8 ), 16) % 100)
            ) % 100 < 9.5 then 'control'
        else 'treatment'
    end as cohort, -- new cohort 
    upsell_user_id, 
    buy_pass_15d, 
    total_trip_dropoff, 
    total_request_dropoff, 
    upsell_session_id,
    origin_hexagon,
    destination_hexagon,
    landing_page_label,
    get_pass_label,
    pass_purchase_label,
    trip_complete_84d,
    trip_complete_per_days_active_84d,
    promo_used_84d,
    trip_x_prc_84d,
    trip_pool_prc_84d,
    trip_pool_per_x_84d,
    session_per_days_active_84d,
    session_request_prc_84d,
    session_background_pre_request_prc_84d,
    has_session_request_84d,
    duration_session_outside_total_prc_84d,
    has_session_without_request_84d,
    payment_cash_trip_prc_84d,
    surge_trip_prc_84d,
    ufp_trip_not_honored_prc_84d,
    ufp_trip_total_prc_84d,
    trip_promo_prc_84d,
    trip_complete_prc_84d,
    trip_rider_cancelled_prc_84d,
    trip_driver_cancelled_prc_84d,
    request_to_trip_prc_84d,
    days_session_request_prc_84d,
    trips_lifetime,
    trip_complete_win7d_potential_84d,
    days_since_trip_first_lifetime,
    trip_complete_win28d_potential_84d,
    fare_total_win7d_potential_84d,
    trip_total_total_84d,
    fare_total_win28d_potential_84d,
    days_since_last_soft_churn_lifetime,
    days_active_84d,
    days_since_last_hard_churn_lifetime,
    session_lt_1m_prc_84d,
    fare_max_p50_84d,
    uber_preferred_score,
    home_city,
    pct_sessions_peak_hour,
    pct_sessions_weekend,
    pct_sessions_surged,
    avg_surge_multiplier,
    p10_surge_multiplier,
    p50_surge_multiplier,
    p90_surge_multiplier,
    tot_potential_saving_subs,
    avg_potential_saving_subs,
    p10_potential_saving_subs,
    p50_potential_saving_subs,
    p90_potential_saving_subs,
    avg_session_eta_last,
    p10_session_eta_last,
    p50_session_eta_last,
    p90_session_eta_last,
    prob_has_promotion,
    prob_has_subscription,
    avg_promotions,
    avg_subscriptions,
    num_subs,
    num_refunds,
    time_since_last_subs,
    time_since_refund,
    accum_upsell_impression,
    accum_landing_page_view,
    accum_landing_page_get_pass,
    pcor_activated,
    pcor_eligible,
    week_of_year,
    weekday_local,
    hour_of_day_local,
    hour_of_week_local,
    path_type,
    is_traveler,
    is_eater,
    x_ufp_fare_before_sub_promotion,
    x_eta,
    x_surge,
    x_estimate_fare_distance_in_miles,
    x_estimate_fare_duration_in_minutes,
    x_promotion,
    x_rsp_multiplier,
    x_rsp_amount,
    x_tolls_surcharges,
    x_predicted_fare_rd,
    x_predicted_saving,
    sr_ufp_fare_before_sub_promotion,
    sr_surge,
    sr_estimate_fare_distance_in_miles,
    sr_estimate_fare_duration_in_minutes,
    sr_promotion,
    sr_rsp_multiplier,
    sr_rsp_amount,
    sr_tolls_surcharges,
    sr_predicted_fare_rd,
    sr_predicted_saving,
    total_ufp_by_path,
    total_ufp_by_origin,
    total_ufp_by_destination,
    pct_ufp_by_path,
    pct_ufp_by_origin,
    pct_ufp_by_destination,
    avg_surged_by_origin,
    num_completed_trips,
    num_non_completed_trips,
    total_trips,
    average_completed_fare,
    average_non_completed_fare,
    average_completed_base_fare,
    average_non_completed_base_fare,
    average_completed_trip_distance_miles,
    average_completed_fare_distance_miles,
    average_completed_fare_duration_minutes,
    c2r,
    num_unique_riders,
    avg_surge_multiplier_on_completed_trips,
    avg_surge_multiplier_on_non_completed_trips,
    pct_surged_complete_trips,
    pct_surged_non_complete_trips,
    avg_completed_eta,
    avg_non_completed_eta,
    pct_trips_on_time,
    pct_completed_trips_with_promotion,
    pct_non_completed_trips_with_promotion,
    pct_completed_trips_commute_hours,
    pct_non_completed_trips_commute_hours,
    pct_completed_trips_weekend,
    pct_non_completed_trips_weekend,
    avg_completed_trip_to_sphere_distance_ratio,
    avg_non_completed_trip_to_sphere_distance_ratio,
    pct_airport_pickup_completed_trips,
    pct_airport_pickup_non_completed_trips,
    pct_airport_dropoff_completed_trips,
    pct_airport_dropoff_non_completed_trips,
    pct_sub_completed_trips,
    pct_sub_non_completed_trips,
    pickup_trip_start_num_completed_trips,
    pickup_trip_start_c2r,
    pickup_trip_start_pct_x_trips,
    pickup_trip_start_num_unique_riders,
    pickup_trip_start_avg_surge_multiplier_on_requests,
    pickup_trip_start_pct_surged_trips,
    pickup_trip_start_avg_eta,
    pickup_trip_start_tot_ufp,
    pickup_trip_start_avg_trip_distance_miles,
    pickup_trip_start_pct_trips_on_time,
    pickup_trip_start_pct_trips_with_promotion,
    pickup_trip_start_pct_trips_commute_hours,
    pickup_trip_start_pct_trips_weekend,
    pickup_trip_start_avg_trip_to_sphere_distance_ratio,
    pickup_trip_start_pct_airport_trips,
    pickup_trip_end_num_completed_trips,
    pickup_trip_end_c2r,
    pickup_trip_end_pct_x_trips,
    pickup_trip_end_num_unique_riders,
    pickup_trip_end_avg_surge_multiplier_on_requests,
    pickup_trip_end_pct_surged_trips,
    pickup_trip_end_avg_eta,
    pickup_trip_end_tot_ufp,
    pickup_trip_end_avg_trip_distance_miles,
    pickup_trip_end_pct_trips_on_time,
    pickup_trip_end_pct_trips_with_promotion,
    pickup_trip_end_pct_trips_commute_hours,
    pickup_trip_end_pct_trips_weekend,
    pickup_trip_end_avg_trip_to_sphere_distance_ratio,
    pickup_trip_end_pct_airport_trips,
    dropoff_trip_start_num_completed_trips,
    dropoff_trip_start_c2r,
    dropoff_trip_start_pct_x_trips,
    dropoff_trip_start_num_unique_riders,
    dropoff_trip_start_avg_surge_multiplier_on_requests,
    dropoff_trip_start_pct_surged_trips,
    dropoff_trip_start_avg_eta,
    dropoff_trip_start_tot_ufp,
    dropoff_trip_start_avg_trip_distance_miles,
    dropoff_trip_start_pct_trips_on_time,
    dropoff_trip_start_pct_trips_with_promotion,
    dropoff_trip_start_pct_trips_commute_hours,
    dropoff_trip_start_pct_trips_weekend,
    dropoff_trip_start_avg_trip_to_sphere_distance_ratio,
    dropoff_trip_start_pct_airport_trips,
    dropoff_trip_end_num_completed_trips,
    dropoff_trip_end_c2r,
    dropoff_trip_end_pct_x_trips,
    dropoff_trip_end_num_unique_riders,
    dropoff_trip_end_avg_surge_multiplier_on_requests,
    dropoff_trip_end_pct_surged_trips,
    dropoff_trip_end_avg_eta,
    dropoff_trip_end_tot_ufp,
    dropoff_trip_end_avg_trip_distance_miles,
    dropoff_trip_end_pct_trips_on_time,
    dropoff_trip_end_pct_trips_with_promotion,
    dropoff_trip_end_pct_trips_commute_hours,
    dropoff_trip_end_pct_trips_weekend,
    dropoff_trip_end_avg_trip_to_sphere_distance_ratio,
    dropoff_trip_end_pct_airport_trips,
    datestr
from 
    grouped 
where city_id in ({cities}) 
), 

all_subs_records as ( 
    SELECT 
        datestr, 
        session_id as session_id, 
        rider_id as bought_user_uuid, 
        city_id, 
        min(ts) as ts 
    FROM subscriptions.uberplus_event_user 
    WHERE 
        datestr between cast((date '{yesterday_ds}' - interval '15' day) as VARCHAR) and cast((date '{yesterday_ds}') as VARCHAR) 
        --date_add('{yesterday_ds}' , -15) and '{yesterday_ds}' 
        AND name in ('uber_pass_tap_purchase', 'subs_purchase_screen_purchase') 
        AND city_id IN (SELECT DISTINCT city_id FROM subscriptions.upsell_city_offer_info) --({cities}) --
    GROUP BY 1, 2, 3, 4 
) 

select 
    case when (a.bought_user_uuid is null) then 0.0 else 1.0 end as buy_pass_15d_all, 
    l.* 
from 
    labels_with_features l 
left join 
    all_subs_records a 
    on 
    l.upsell_user_id = a.bought_user_uuid 
""".format(yesterday_ds=yesterday_ds, city_ids=cities) 

def hscls_model_data(start_date, end_date, city_ids, bucket_cap): 
    return """
with left_table as ( 
    select 
        datestr, 
        rider_uuid, 
        strategy_name, 
        is_treatment, 
        cohort, 
        targeting_model, 
        ((FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(rider_uuid)))), 1,  8 ), 16) % 100) * (96*96*96 % 100) + (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(rider_uuid)))), 9,  8 ), 16) % 100) * (96*96 % 100) + (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(rider_uuid)))), 17, 8 ), 16) % 100) * (96 % 100) + (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(rider_uuid)))), 25, 8 ), 16) % 100)) % 100 as randbucket 
    from personalization.sapphire_rider_treatment 
    where 
        datestr > '{start_date}' 
        and datestr < '{end_date}' 
        and proposal_city_id in ({city_ids}) 
), 

features as ( 
    select 
        uuid, 
        churns_hard_lifetime, 
        days_active_lifetime, 
        days_since_trip_first_lifetime, 
        fare_lifetime, 
        days_active_84d, 
        trip_pool_matched_avg_84d, 
        fare_promo_total_avg_84d, 
        fare_total_avg_84d, 
        ata_trip_max_avg_84d, 
        eta_trip_max_avg_84d, 
        rating_2rider_total_avg_84d, 
        surge_trip_avg_84d, 
        fare_total_win7d_potential_84d, 
        trip_complete_win7d_potential_84d, 
        trip_total_win7d_potential_84d, 
        fare_total_win28d_potential_84d, 
        trip_complete_win28d_potential_84d, 
        trip_total_win28d_potential_84d, 
        datestr 
    from 
        gi_models.rider_flib 
    where 
        datestr > '{start_date}' 
        and datestr < '{end_date}' 
        and city_id in ({city_ids}) 
) 

, joined as ( 
    select 
        l.datestr, 
        l.rider_uuid as  rider_uuid, 
        l.strategy_name, 
        l.is_treatment, 
        l.cohort, 
        l.targeting_model, 
        l.randbucket, 
        f.churns_hard_lifetime, 
        f.days_active_lifetime, 
        f.days_since_trip_first_lifetime, 
        f.fare_lifetime, 
        f.days_active_84d, 
        f.trip_pool_matched_avg_84d, 
        f.fare_promo_total_avg_84d, 
        f.fare_total_avg_84d, 
        f.ata_trip_max_avg_84d, 
        f.eta_trip_max_avg_84d, 
        f.rating_2rider_total_avg_84d, 
        f.surge_trip_avg_84d, 
        f.fare_total_win7d_potential_84d, 
        f.trip_complete_win7d_potential_84d, 
        f.trip_total_win7d_potential_84d, 
        f.fare_total_win28d_potential_84d, 
        f.trip_complete_win28d_potential_84d, 
        f.trip_total_win28d_potential_84d 
    from 
        left_table l 
        left join 
        features f 
    on l.datestr = f.datestr 
    and l.rider_uuid = f.uuid 
) 

, labels as ( 
    select 
        datestr,
        rider_uuid, 
        trip_most_freq_city_id, 
        num_trips, 
        gross_bookings_usd, 
        variable_contribution_usd, 
        net_billings_usd
    from 
        personalization.riders_weekly_stats_core_metrics 
) 

, joined_labels as ( 
    select 
        j.datestr as datestr, 
        j.rider_uuid as rider_uuid, 
        j.strategy_name, 
        j.is_treatment, 
        j.cohort, 
        j.targeting_model, 
        j.randbucket, 
        j.churns_hard_lifetime, 
        j.days_active_lifetime, 
        j.days_since_trip_first_lifetime, 
        j.fare_lifetime, 
        j.days_active_84d, 
        j.trip_pool_matched_avg_84d, 
        j.fare_promo_total_avg_84d, 
        j.fare_total_avg_84d, 
        j.ata_trip_max_avg_84d, 
        j.eta_trip_max_avg_84d, 
        j.rating_2rider_total_avg_84d, 
        j.surge_trip_avg_84d, 
        j.fare_total_win7d_potential_84d, 
        j.trip_complete_win7d_potential_84d, 
        j.trip_total_win7d_potential_84d, 
        j.fare_total_win28d_potential_84d, 
        j.trip_complete_win28d_potential_84d, 
        j.trip_total_win28d_potential_84d, 
        l.trip_most_freq_city_id, 
        l.num_trips, 
        l.gross_bookings_usd, 
        l.variable_contribution_usd, 
        l.net_billings_usd 
    from 
    joined j 
    left join 
    labels l 
    on j.rider_uuid = l.rider_uuid 
    and j.datestr = l.datestr 
) 

, selected as ( 
select * from 
joined_labels 
where randbucket <= {bucket_cap} 
) 

select * 
from 
selected 
""".format(start_date=start_date, end_date=end_date, city_ids=city_ids, bucket_cap=bucket_cap) 

def hscls_model_data_dualcap(start_date, end_date, city_ids, bucket_cap1, bucket_cap2): 
    return """
with left_table as ( 
    select 
        datestr, 
        rider_uuid, 
        strategy_name, 
        is_treatment, 
        cohort, 
        targeting_model, 
        ((FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(rider_uuid)))), 1,  8 ), 16) % 100) * (96*96*96 % 100) + (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(rider_uuid)))), 9,  8 ), 16) % 100) * (96*96 % 100) + (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(rider_uuid)))), 17, 8 ), 16) % 100) * (96 % 100) + (FROM_BASE( SUBSTR( LOWER(TO_HEX(MD5(TO_UTF8(rider_uuid)))), 25, 8 ), 16) % 100)) % 100 as randbucket 
    from personalization.sapphire_rider_treatment 
    where 
        datestr > '{start_date}' 
        and datestr < '{end_date}' 
        and proposal_city_id in ({city_ids}) 
), 

features as ( 
    select 
        uuid, 
        churns_hard_lifetime, 
        days_active_lifetime, 
        days_since_trip_first_lifetime, 
        fare_lifetime, 
        days_active_84d, 
        trip_pool_matched_avg_84d, 
        fare_promo_total_avg_84d, 
        fare_total_avg_84d, 
        ata_trip_max_avg_84d, 
        eta_trip_max_avg_84d, 
        rating_2rider_total_avg_84d, 
        surge_trip_avg_84d, 
        fare_total_win7d_potential_84d, 
        trip_complete_win7d_potential_84d, 
        trip_total_win7d_potential_84d, 
        fare_total_win28d_potential_84d, 
        trip_complete_win28d_potential_84d, 
        trip_total_win28d_potential_84d, 
        datestr 
    from 
        gi_models.rider_flib 
    where 
        datestr > '{start_date}' 
        and datestr < '{end_date}' 
        and city_id in ({city_ids}) 
) 

, joined as ( 
    select 
        l.datestr, 
        l.rider_uuid as  rider_uuid, 
        l.strategy_name, 
        l.is_treatment, 
        l.cohort, 
        l.targeting_model, 
        l.randbucket, 
        f.churns_hard_lifetime, 
        f.days_active_lifetime, 
        f.days_since_trip_first_lifetime, 
        f.fare_lifetime, 
        f.days_active_84d, 
        f.trip_pool_matched_avg_84d, 
        f.fare_promo_total_avg_84d, 
        f.fare_total_avg_84d, 
        f.ata_trip_max_avg_84d, 
        f.eta_trip_max_avg_84d, 
        f.rating_2rider_total_avg_84d, 
        f.surge_trip_avg_84d, 
        f.fare_total_win7d_potential_84d, 
        f.trip_complete_win7d_potential_84d, 
        f.trip_total_win7d_potential_84d, 
        f.fare_total_win28d_potential_84d, 
        f.trip_complete_win28d_potential_84d, 
        f.trip_total_win28d_potential_84d 
    from 
        left_table l 
        left join 
        features f 
    on l.datestr = f.datestr 
    and l.rider_uuid = f.uuid 
) 

, labels as ( 
    select 
        datestr,
        rider_uuid, 
        trip_most_freq_city_id, 
        num_trips, 
        gross_bookings_usd, 
        variable_contribution_usd, 
        net_billings_usd
    from 
        personalization.riders_weekly_stats_core_metrics 
) 

, joined_labels as ( 
    select 
        j.datestr as datestr, 
        j.rider_uuid as rider_uuid, 
        j.strategy_name, 
        j.is_treatment, 
        j.cohort, 
        j.targeting_model, 
        j.randbucket, 
        j.churns_hard_lifetime, 
        j.days_active_lifetime, 
        j.days_since_trip_first_lifetime, 
        j.fare_lifetime, 
        j.days_active_84d, 
        j.trip_pool_matched_avg_84d, 
        j.fare_promo_total_avg_84d, 
        j.fare_total_avg_84d, 
        j.ata_trip_max_avg_84d, 
        j.eta_trip_max_avg_84d, 
        j.rating_2rider_total_avg_84d, 
        j.surge_trip_avg_84d, 
        j.fare_total_win7d_potential_84d, 
        j.trip_complete_win7d_potential_84d, 
        j.trip_total_win7d_potential_84d, 
        j.fare_total_win28d_potential_84d, 
        j.trip_complete_win28d_potential_84d, 
        j.trip_total_win28d_potential_84d, 
        l.trip_most_freq_city_id, 
        l.num_trips, 
        l.gross_bookings_usd, 
        l.variable_contribution_usd, 
        l.net_billings_usd 
    from 
    joined j 
    left join 
    labels l 
    on j.rider_uuid = l.rider_uuid 
    and j.datestr = l.datestr 
) 

, selected as ( 
select * from 
joined_labels 
where randbucket >= {bucket_cap1} 
and randbucket <= {bucket_cap2} 
) 

select * 
from 
selected 
""".format(start_date=start_date, end_date=end_date, city_ids=city_ids, bucket_cap1=bucket_cap1, bucket_cap2=bucket_cap2) 
