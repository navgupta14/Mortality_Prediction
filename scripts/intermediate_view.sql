create view intermediatefeatures_pvairamani3 as
(
    SELECT adm.hadm_id, admittime, dischtime, adm.deathtime, pat.dod, pat.gender, extract(YEAR FROM admittime) - extract(YEAR FROM dob) AS age, pat.subject_id, pat.expire_flag
    -- integer which is 1 for the most recent hospital admission
    , ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY admittime DESC) AS mostrecent
    FROM admissions adm
    INNER JOIN patients pat
    ON adm.subject_id = pat.subject_id
    -- filter out organ donor accounts
    WHERE lower(diagnosis) NOT LIKE '%organ donor%'
    -- at least 15 years old
    AND extract(YEAR FROM admittime) - extract(YEAR FROM dob) > 15
    -- filter that removes hospital admissions with no corresponding ICU data
    AND HAS_CHARTEVENTS_DATA = 1
);
