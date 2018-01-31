select frn.*, fli.*,

case when purpose = 'Internet access service that includes a connection from any applicant site directly to the Internet Service Provider'
then 'ias_includes_connection'
when purpose = 'Internet access service with no circuit (data circuit to ISP state/regional network is billed separately)'
then 'ias_no_circuit'
when purpose = 'Data connection(s) for an applicant’s hub site to an Internet Service Provider or state/regional network where Internet access service is billed separately'
then 'data_connect_hub'
when purpose = 'Data Connection between two or more sites entirely within the applicant’s network'
then 'data_connect_2ormore'
when purpose = 'Backbone circuit for consortium that provides connectivity between aggregation points or other non-user facilities'
then 'backbone'
else 'Unknown' end as purpose_adj,

case when upload_speed_units ilike '%K'
then upload_speed::numeric/1000
when upload_speed_units ilike '%G%'
then upload_speed::numeric*1000
else upload_speed::numeric end as upload_speed_mbps,

case when download_speed_units ilike '%K'
then download_speed::numeric/1000
when download_speed_units ilike '%G%'
then download_speed::numeric*1000
else download_speed::numeric end as download_speed_mbps,

case
when contract_expiry_date::date <= '2018-06-30'
then 1
when contract_expiry_date::date <= '2019-06-30'
then 2
when contract_expiry_date::date <= '2020-06-30'
then 3
when contract_expiry_date::date <= '2021-06-30'
then 4
when contract_expiry_date::date <= '2022-06-30'
then 5
when contract_expiry_date::date <= '2023-06-30'
then 6
when contract_expiry_date::date <= '2024-06-30'
then 7
when contract_expiry_date::date <= '2025-06-30'
then 8
end as contract_end_time,

case when frn_number_from_the_previous_year is not null 
then 'Yes' 
else 'No' 
end as frn_previous_year_exists

from fy2017.frns frn
left join fy2017.frn_line_items fli
on frn.frn = fli.frn

where function not in (
    'Miscellaneous',
    'Cabinets', 
    'Cabling', 
    'Conduit',
    'Connectors/Couplers', 
    'Patch Panels', 
    'Routers', 
    'Switches', 
    'UPS')

and service_type in ('Data Transmission and/or Internet Access')