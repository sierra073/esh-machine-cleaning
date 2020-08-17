select frn.*,
fli.basic_firewall_protection,
fli.burstable_speed,
fli.burstable_speed_units,
fli.connected_directly_to_school_library_or_nif,
fli.connection_supports_school_library_or_nif,
fli.connection_used_by,
fli.download_speed,
fli.download_speed_units,
fli.function,
fli.is_installation_included_in_price,
fli.lease_or_non_purchase_agreement,
fli.line_item,
fli.make,
fli.model,
fli.monthly_quantity,
fli.monthly_recurring_unit_costs,
fli.monthly_recurring_unit_eligible_costs,
fli.monthly_recurring_unit_ineligible_costs,
fli.months_of_service,
fli.one_time_eligible_unit_costs,
fli.one_time_ineligible_unit_costs,
fli.one_time_quantity,
fli.one_time_unit_costs,
fli.other_manufacture,
fli.pre_discount_extended_eligible_line_item_cost,
fli.purpose,
fli.total_eligible_one_time_costs,
fli.total_eligible_recurring_costs,
fli.total_monthly_eligible_recurring_costs,
fli.type_of_product,
fli.unit,
fli.upload_speed,
fli.upload_speed_units,

replace(line_item::varchar,'.', '-') AS frn_adjusted,

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
when contract_expiration_date::date <= '2020-06-30'
then 1
when contract_expiration_date::date <= '2021-06-30'
then 2
when contract_expiration_date::date <= '2022-06-30'
then 3
when contract_expiration_date::date <= '2023-06-30'
then 4
when contract_expiration_date::date <= '2024-06-30'
then 5
when contract_expiration_date::date <= '2025-06-30'
then 6
when contract_expiration_date::date <= '2026-06-30'
then 7
when contract_expiration_date::date <= '2027-06-30'
then 8
end as contract_end_time,

case when frn_number_from_the_previous_year is not null
then 'Yes'
else 'No'
end as frn_previous_year_exists

from ing.fy2019_frns frn --check these table sources
left join ing.fy2019_frn_line_items fli --check these table sources
on frn.frn = fli.frn --check these table sources

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

and (fiber_sub_type is null)

and service_type in ('Data Transmission and/or Internet Access')
