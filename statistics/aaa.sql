SELECT
  halo_lm,
  CASE
    WHEN true_redshift_halo < 0.105 THEN 1 
    WHEN true_redshift_halo < 0.11  THEN 2
    WHEN true_redshift_halo < 0.115 THEN 3
    WHEN true_redshift_halo < 0.12  THEN 4
    WHEN true_redshift_halo < 0.125 THEN 5
    WHEN true_redshift_halo < 0.13  THEN 6
    WHEN true_redshift_halo < 0.135 THEN 7
    WHEN true_redshift_halo < 0.14  THEN 8
    WHEN true_redshift_halo < 0.145 THEN 9
    WHEN true_redshift_halo < 0.15  THEN 10
    WHEN true_redshift_halo < 0.155 THEN 11
    WHEN true_redshift_halo < 0.16  THEN 12
    WHEN true_redshift_halo < 0.165 THEN 13
    WHEN true_redshift_halo < 0.17  THEN 14
    WHEN true_redshift_halo < 0.175 THEN 15
    WHEN true_redshift_halo < 0.18  THEN 16
    WHEN true_redshift_halo < 0.185 THEN 17
    WHEN true_redshift_halo < 0.19  THEN 18
    WHEN true_redshift_halo < 0.195 THEN 19
    WHEN true_redshift_halo < 0.2   THEN 20
    WHEN true_redshift_halo < 0.205 THEN 21
    WHEN true_redshift_halo < 0.21  THEN 22
    WHEN true_redshift_halo < 0.215 THEN 23
    WHEN true_redshift_halo < 0.22  THEN 24
    WHEN true_redshift_halo < 0.225 THEN 25
    WHEN true_redshift_halo < 0.23  THEN 26
    WHEN true_redshift_halo < 0.235 THEN 27
    WHEN true_redshift_halo < 0.24  THEN 28
    WHEN true_redshift_halo < 0.245 THEN 29
    WHEN true_redshift_halo < 0.25  THEN 30
    WHEN true_redshift_halo < 0.255 THEN 31
    WHEN true_redshift_halo < 0.26  THEN 32
    WHEN true_redshift_halo < 0.265 THEN 33
    WHEN true_redshift_halo < 0.27  THEN 34
    WHEN true_redshift_halo < 0.275 THEN 35
    WHEN true_redshift_halo < 0.28  THEN 36
    WHEN true_redshift_halo < 0.285 THEN 37
    WHEN true_redshift_halo < 0.29  THEN 38
    WHEN true_redshift_halo < 0.295 THEN 39
    WHEN true_redshift_halo < 0.3   THEN 40
    WHEN true_redshift_halo < 0.305 THEN 41
    WHEN true_redshift_halo < 0.31  THEN 42
    WHEN true_redshift_halo < 0.315 THEN 43
    WHEN true_redshift_halo < 0.32  THEN 44
    WHEN true_redshift_halo < 0.325 THEN 45
    WHEN true_redshift_halo < 0.33  THEN 46
    WHEN true_redshift_halo < 0.335 THEN 47
    WHEN true_redshift_halo < 0.34  THEN 48
    WHEN true_redshift_halo < 0.345 THEN 49
    WHEN true_redshift_halo < 0.35  THEN 50
    WHEN true_redshift_halo < 0.355 THEN 51
    WHEN true_redshift_halo < 0.36  THEN 52
    WHEN true_redshift_halo < 0.365 THEN 53
    WHEN true_redshift_halo < 0.37  THEN 54
    WHEN true_redshift_halo < 0.375 THEN 55
    WHEN true_redshift_halo < 0.38  THEN 56
    WHEN true_redshift_halo < 0.385 THEN 57
    WHEN true_redshift_halo < 0.39  THEN 58
    WHEN true_redshift_halo < 0.395 THEN 59
    WHEN true_redshift_halo < 0.4   THEN 60
    WHEN true_redshift_halo < 0.405 THEN 61
    WHEN true_redshift_halo < 0.41  THEN 62
    WHEN true_redshift_halo < 0.415 THEN 63
    WHEN true_redshift_halo < 0.42  THEN 64
    WHEN true_redshift_halo < 0.425 THEN 65
    WHEN true_redshift_halo < 0.43  THEN 66
    WHEN true_redshift_halo < 0.435 THEN 67
    WHEN true_redshift_halo < 0.44  THEN 68
    WHEN true_redshift_halo < 0.445 THEN 69
    WHEN true_redshift_halo < 0.45  THEN 70
    WHEN true_redshift_halo < 0.455 THEN 71
    WHEN true_redshift_halo < 0.46  THEN 72
    WHEN true_redshift_halo < 0.465 THEN 73
    WHEN true_redshift_halo < 0.47  THEN 74
    WHEN true_redshift_halo < 0.475 THEN 75
    WHEN true_redshift_halo < 0.48  THEN 76
    WHEN true_redshift_halo < 0.485 THEN 77
    WHEN true_redshift_halo < 0.49  THEN 78
    WHEN true_redshift_halo < 0.495 THEN 79
    WHEN true_redshift_halo < 0.5   THEN 80
    WHEN true_redshift_halo < 0.505 THEN 81
    WHEN true_redshift_halo < 0.51  THEN 82
    WHEN true_redshift_halo < 0.515 THEN 83
    WHEN true_redshift_halo < 0.52  THEN 84
    WHEN true_redshift_halo < 0.525 THEN 85
    WHEN true_redshift_halo < 0.53  THEN 86
    WHEN true_redshift_halo < 0.535 THEN 87
    WHEN true_redshift_halo < 0.54  THEN 88
    WHEN true_redshift_halo < 0.545 THEN 89
    WHEN true_redshift_halo < 0.55  THEN 90
    WHEN true_redshift_halo < 0.555 THEN 91
    WHEN true_redshift_halo < 0.56  THEN 92
    WHEN true_redshift_halo < 0.565 THEN 93
    WHEN true_redshift_halo < 0.57  THEN 94
    WHEN true_redshift_halo < 0.575 THEN 95
    WHEN true_redshift_halo < 0.58  THEN 96
    WHEN true_redshift_halo < 0.585 THEN 97
    WHEN true_redshift_halo < 0.59  THEN 98
    WHEN true_redshift_halo < 0.595 THEN 99
    WHEN true_redshift_halo < 0.6   THEN 100
    ELSE 0
  END AS z_cgal_bin
FROM (
  SELECT halo_lm, true_redshift_halo
  FROM l768_mg_v1_1_lensing_c_zphot 
) AS t
