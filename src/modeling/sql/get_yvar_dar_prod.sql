SELECT DISTINCT
replace(li.frn_complete::varchar,'.', '-') AS frn_adjusted,
-- sr.yvar gets replaced depending on user input
li.yvar

FROM ps.districts_line_items dli
JOIN ps.districts d
ON 	dli.district_id = d.district_id
AND dli.funding_year = d.funding_year

JOIN ps.line_items li
ON dli.line_item_id = li.line_item_id
JOIN
(SELECT frn_complete, count(line_item_id) as cnt
FROM ps.line_items li
WHERE li.funding_year = 2019
GROUP BY 1) cnts
ON li.frn_complete = cnts.frn_complete

WHERE li.exclude_labels = 0
AND li.dirty_labels = 0
AND cnts.cnt = 1
