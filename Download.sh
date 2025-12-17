aws s3 sync --no-sign-request "s3://smn-ar-wrf/DATA/WRF/DET/2024" images/2024/ \
  --exclude "*10M*.nc" \
  --exclude "*24H*.nc" \
  --only-show-errors 

