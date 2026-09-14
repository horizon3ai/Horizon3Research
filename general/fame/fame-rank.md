# Objective prominence ranking (training-data prominence)

Equation + weights: see `fame_rank.py`. `fame_index` 0-100; NVD refs excluded by design. Tiers: obscure <34, mid 34-61, famous >=62.

Scored 45 CVEs; 0 pending proxy data.

| rank | fame_index | new_tier | old_tier | flip | CVE | app | era | cvss |
|---:|---:|---|---|---|---|---|---:|---:|
| 1 | 100.0 | famous | landmark | landmark->famous | MS17-010 | Windows SMBv1 (EternalBlue) | 2017 | 8.1 |
| 2 | 100.0 | famous | landmark | landmark->famous | CVE-2021-44228 | Apache Log4j (Log4Shell) | 2021 | 10.0 |
| 3 | 100.0 | famous | famous |  | CVE-2022-22965 | Spring Framework 5.3.17 | 2022 | 9.8 |
| 4 | 89.2 | famous | landmark | landmark->famous | CVE-2014-6271 | Apache httpd 2.4.18 | 2014 | 9.8 |
| 5 | 87.3 | famous | landmark | landmark->famous | CVE-2014-0160 | OpenSSL (Heartbleed) | 2014 | 7.5 |
| 6 | 84.6 | famous | famous |  | CVE-2019-19781 | Citrix ADC | 2019 | 9.8 |
| 7 | 81.8 | famous | mid | mid->famous | CVE-2022-22947 | Spring Cloud Gateway 3.1.0 | 2022 | 10.0 |
| 8 | 79.9 | famous | famous |  | CVE-2023-22527 | Atlassian Confluence 8.5.3 | 2023 | 9.8 |
| 9 | 79.5 | famous | famous |  | CVE-2017-7494 | Samba 4.6.3 | 2017 | 9.8 |
| 10 | 73.7 | famous | mid | mid->famous | CVE-2021-43798 | Grafana 8.2.6 | 2021 | 7.5 |
| 11 | 73.1 | famous | high | high->famous | CVE-2025-24813 | Apache Tomcat 9.0.97 | 2025 | 9.8 |
| 12 | 72.8 | famous | high | high->famous | CVE-2024-23897 | Jenkins 2.441 | 2024 | 9.8 |
| 13 | 71.3 | famous | famous |  | CVE-2020-14882 | Oracle WebLogic 12.2.1.3 | 2020 | 9.8 |
| 14 | 71.3 | famous | famous |  | CVE-2023-46604 | Apache ActiveMQ 5.17.3 | 2023 | 9.8 |
| 15 | 69.8 | famous | mid | mid->famous | CVE-2021-22205 | GitLab CE 13.10.3 | 2021 | 9.8 |
| 16 | 68.4 | famous | high | high->famous | CVE-2025-32433 | Erlang/OTP 27.3.2 (SSH) | 2025 | 10.0 |
| 17 | 68.2 | famous | famous |  | CVE-2022-0543 | Redis 5.0.7 | 2022 | 10.0 |
| 18 | 67.5 | famous | high | high->famous | CVE-2025-1974 | ingress-nginx 1.9.5 | 2025 | 9.8 |
| 19 | 66.3 | famous | mid | mid->famous | CVE-2023-23752 | Joomla 4.2.7 | 2023 | 5.3 |
| 20 | 65.9 | famous | high | high->famous | CVE-2023-42793 | JetBrains TeamCity 2023.05.3 | 2023 | 9.8 |
| 21 | 65.7 | famous | mid | mid->famous | CVE-2011-2523 | vsftpd 2.3.4 | 2011 | 9.8 |
| 22 | 65.2 | famous | high | high->famous | CVE-2023-21839 | Oracle WebLogic 12.2.1.3 | 2023 | 7.5 |
| 23 | 64.7 | famous | mid | mid->famous | CVE-2023-27350 | PaperCut NG 22.0.5 | 2023 | 9.8 |
| 24 | 64.7 | famous | mid | mid->famous | CVE-2024-36401 | GeoServer 2.23.2 | 2024 | 9.8 |
| 25 | 64.7 | famous | mid | mid->famous | CVE-2023-32315 | Openfire 4.7.4 | 2023 | 8.6 |
| 26 | 63.6 | famous | mid | mid->famous | CVE-2023-38646 | Metabase 0.46.6 | 2023 | 9.8 |
| 27 | 62.4 | famous | obscure | obscure->famous | CVE-2022-46169 | Cacti 1.2.22 | 2022 | 9.8 |
| 28 | 53.1 | mid | mid |  | CVE-2019-7609 | Kibana 6.5.4 | 2019 | 10.0 |
| 29 | 52.6 | mid | obscure | obscure->mid | CVE-2017-12617 | Apache Tomcat 9.0.0 | 2017 | 8.1 |
| 30 | 50.1 | mid | mid |  | CVE-2025-32432 | Craft CMS 5.6.16 | 2025 | 10.0 |
| 31 | 49.8 | mid | mid |  | CVE-2019-0193 | Apache Solr 8.1.1 | 2019 | 8.8 |
| 32 | 48.9 | mid | obscure | obscure->mid | CVE-2017-17562 | GoAhead 3.6.4 | 2017 | 9.8 |
| 33 | 48.9 | mid | mid |  | CVE-2015-1427 | Elasticsearch 1.4.2 | 2015 | 9.8 |
| 34 | 48.9 | mid | low | low->mid | CVE-2024-27348 | Apache HugeGraph 1.2.0 | 2024 | 9.8 |
| 35 | 43.2 | mid | mid |  | CVE-2018-1000861 | Jenkins 2.138 | 2018 | 9.8 |
| 36 | 42.3 | mid | obscure | obscure->mid | CVE-2024-4956 | Nexus Repository 3.68.0 | 2024 | 7.5 |
| 37 | 35.7 | mid | mid |  | CVE-2023-25194 | Apache Druid 25.0.0 | 2023 | 8.8 |
| 38 | 32.5 | obscure | mid | mid->obscure | CVE-2024-45195 | Apache OFBiz 18.12.15 | 2024 | 9.8 |
| 39 | 27.8 | obscure | obscure |  | CVE-2010-20103 | ProFTPD 1.3.3c | 2010 | 9.8 |
| 40 | 27.8 | obscure | obscure |  | CVE-2018-25270 | ThinkPHP 5.0.23 | 2018 | 9.8 |
| 41 | 27.0 | obscure | mid | mid->obscure | CVE-2025-2945 | pgAdmin 4 9.1 | 2025 | 9.9 |
| 42 | 19.6 | obscure | low | low->obscure | CVE-2023-25826 | OpenTSDB 2.4.1 | 2023 | 9.8 |
| 43 | 14.6 | obscure | obscure |  | CVE-2024-21644 | pyLoad 0.5.0b3.dev31 | 2024 | 7.5 |
| 44 | 14.6 | obscure | obscure |  | CVE-2023-33365 | Suprema BioStar 2 2.9.0 | 2023 | 7.5 |
| 45 | 14.6 | obscure | obscure |  | CVE-2024-47769 | IDURAR ERP/CRM 4.1.0 | 2024 | 7.5 |
