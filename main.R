# libraries ====================================================================
library(tidymodels)
library(readr)
library(broom.mixed)
library(dotwhisker)

# build a model ================================================================

urchins <-
  # Data were assembled for a tutorial
  # at https://www.flutterbys.com.au/stats/tut/tut7.5a.html
  read_csv("https://tidymodels.org/start/models/urchins.csv") |>
  # Change the names to be a little more verbose
  setNames(c("food_regime", "initial_volume", "width")) |>
  # Factors are very helpful for modeling, so we convert one column
  mutate(
    food_regime = factor(food_regime, levels = c("Initial", "Low", "High"))
  )

ggplot(
  urchins,
  aes(
    x = initial_volume,
    y = width,
    group = food_regime,
    col = food_regime
  )
) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  scale_color_viridis_d(option = "plasma", end = .7)

lm_mod <-
  linear_reg() %>%
  set_engine("lm")

(lm_fit <-
  lm_mod %>%
  fit(width ~ initial_volume * food_regime, data = urchins))

tidy(lm_fit)

tidy(lm_fit) %>%
  dwplot(
    dot_args = list(size = 2, color = "black"),
    whisker_args = list(color = "black"),
    vline = geom_vline(xintercept = 0, colour = "grey50", linetype = 2)
  )

new_points <- expand.grid(
  initial_volume = 20,
  food_regime = c("Initial", "Low", "High")
)

(mean_pred <- predict(lm_fit, new_data = new_points))

(conf_int_pred <- predict(lm_fit,
  new_data = new_points,
  type = "conf_int"
))

# Now combine:
plot_data <-
  new_points %>%
  bind_cols(mean_pred) %>%
  bind_cols(conf_int_pred)

# and plot:
ggplot(plot_data, aes(x = food_regime)) +
  geom_point(aes(y = .pred)) +
  geom_errorbar(aes(
    ymin = .pred_lower,
    ymax = .pred_upper
  ),
  width = .2
  ) +
  labs(y = "urchin size")

# Preprocess data with recipies ================================================
library(nycflights13)
library(skimr)

set.seed(123)

flight_data <-
  flights %>%
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = lubridate::as_date(time_hour)
  ) %>%
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>%
  # Only retain the specific columns we will use
  select(
    dep_time, flight, origin, dest, air_time, distance,
    carrier, date, arr_delay, time_hour
  ) %>%
  # Exclude missing data
  na.omit() %>%
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor)

flight_data %>%
  count(arr_delay) %>%
  mutate(prop = n / sum(n))

glimpse(flight_data)

flight_data %>%
  skimr::skim(dest, carrier)

# data splitting
# Fix the random numbers by setting the seed
# This enables the analysis to be reproducible when random numbers are used
set.seed(222)
# Put 3/4 of the data into the training set
data_split <- initial_split(flight_data, prop = 3 / 4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data <- testing(data_split)

# create recipe and roles
flights_rec <-
  recipe(arr_delay ~ ., data = train_data)

flights_rec <-
  recipe(arr_delay ~ ., data = train_data) %>%
  update_role(flight, time_hour, new_role = "ID")

summary(flights_rec)

flights_rec <-
  recipe(arr_delay ~ ., data = train_data) %>%
  update_role(flight, time_hour, new_role = "ID") %>%
  step_date(date, features = c("dow", "month")) %>%
  step_holiday(date,
    holidays = timeDate::listHolidays("US"),
    keep_original_cols = FALSE
  )
flights_rec <-
  recipe(arr_delay ~ ., data = train_data) %>%
  update_role(flight, time_hour, new_role = "ID") %>%
  step_date(date, features = c("dow", "month")) %>%
  step_holiday(date,
    holidays = timeDate::listHolidays("US"),
    keep_original_cols = FALSE
  ) %>%
  step_dummy(all_nominal_predictors())

flights_rec <-
  recipe(arr_delay ~ ., data = train_data) %>%
  update_role(flight, time_hour, new_role = "ID") %>%
  step_date(date, features = c("dow", "month")) %>%
  step_holiday(date,
    holidays = timeDate::listHolidays("US"),
    keep_original_cols = FALSE
  ) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

flights_rec <-
  recipe(arr_delay ~ ., data = train_data) %>%
  update_role(flight, time_hour, new_role = "ID") %>%
  step_date(date, features = c("dow", "month")) %>%
  step_holiday(date,
    holidays = timeDate::listHolidays("US"),
    keep_original_cols = FALSE
  ) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

# fit a model with a recipe
lr_mod <-
  logistic_reg() %>%
  set_engine("glm")

(flights_wflow <-
  workflow() %>%
  add_model(lr_mod) %>%
  add_recipe(flights_rec))

flights_fit <-
  flights_wflow %>%
  fit(data = train_data)

flights_fit %>%
  extract_fit_parsnip() %>%
  tidy()

# use a traind worklflow to predict
predict(flights_fit, test_data)

flights_aug <-
  augment(flights_fit, test_data)

# The data look like:
flights_aug %>%
  select(arr_delay, time_hour, flight, .pred_class, .pred_on_time)

flights_aug %>%
  roc_curve(truth = arr_delay, .pred_late) %>%
  autoplot()

flights_aug %>%
  roc_auc(truth = arr_delay, .pred_late)