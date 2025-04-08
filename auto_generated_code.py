
# AI Response:
def process_data(data, config):
    """
    This function takes data and a configuration dictionary as input,
    performs a series of transformations and validations based on the
    configuration, and returns a processed version of the data.  It handles
    multiple data types and scenarios, including error handling and logging.
    """

    processed_data = []
    errors = []
    log_messages = []

    # --- Data Type Validation ---
    data_type = config.get("data_type", "unknown")  # Default to unknown if not specified
    log_messages.append(f"Data type detected: {data_type}")

    if data_type == "list":
        if not isinstance(data, list):
            errors.append("Error: Input data is not a list as specified in config.")
            return None, errors, log_messages # Return early if fundamental type is wrong

    elif data_type == "dictionary":
        if not isinstance(data, dict):
            errors.append("Error: Input data is not a dictionary as specified in config.")
            return None, errors, log_messages # Return early if fundamental type is wrong

    elif data_type == "string":
        if not isinstance(data, str):
            errors.append("Error: Input data is not a string as specified in config.")
            return None, errors, log_messages

    elif data_type == "numeric":
        try:
            float(data) # Test if convertable to a number
        except (TypeError, ValueError):
             errors.append("Error: Input data is not numeric as specified in config.")
             return None, errors, log_messages

    else:
        log_messages.append("Warning: Unknown data type specified in config.")

    # --- List Processing ---
    if data_type == "list":
        for item in data:
            try:
                # --- Basic Validation ---
                if config.get("validate_min_length") and len(str(item)) < config["validate_min_length"]:
                    errors.append(f"Error: Item '{item}' is shorter than minimum length {config['validate_min_length']}.")
                    continue  # Skip to the next item

                if config.get("validate_max_length") and len(str(item)) > config["validate_max_length"]:
                     errors.append(f"Error: Item '{item}' is longer than maximum length {config['validate_max_length']}.")
                     continue  # Skip to the next item

                # --- Transformation ---
                transformed_item = item
                if config.get("uppercase"):
                    transformed_item = str(item).upper()
                if config.get("lowercase"):
                    transformed_item = str(item).lower()
                if config.get("add_prefix"):
                    transformed_item = config["add_prefix"] + str(item)
                if config.get("add_suffix"):
                    transformed_item = str(item) + config["add_suffix"]

                # --- Data Cleaning ---
                if config.get("remove_whitespace"):
                    transformed_item = str(transformed_item).strip()

                processed_data.append(transformed_item)

            except Exception as e:
                errors.append(f"Error processing item '{item}': {e}")
                log_messages.append(f"Detailed error: {e}")

    # --- Dictionary Processing ---
    elif data_type == "dictionary":
        for key, value in data.items():
            try:
                # --- Key Validation ---
                if config.get("validate_key_prefix") and not str(key).startswith(config["validate_key_prefix"]):
                    errors.append(f"Error: Key '{key}' does not start with prefix '{config['validate_key_prefix']}'.")
                    continue

                # --- Value Transformation ---
                transformed_value = value
                if config.get("value_uppercase"):
                    transformed_value = str(value).upper()
                if config.get("value_add_prefix"):
                    transformed_value = config["value_add_prefix"] + str(value)

                processed_data.append({key: transformed_value})

            except Exception as e:
                errors.append(f"Error processing key '{key}': {e}")
                log_messages.append(f"Detailed error: {e}")

    # --- String Processing ---
    elif data_type == "string":
        try:
            transformed_data = data
            if config.get("string_replace"):
                 transformed_data = data.replace(config["string_replace"]["old"], config["string_replace"]["new"])
            processed_data = transformed_data
        except Exception as e:
            errors.append(f"Error processing string: {e}")
            log_messages.append(f"Detailed string error: {e}")

    # --- Numeric Processing ---
    elif data_type == "numeric":
        try:
            numeric_data = float(data)
            if config.get("numeric_multiply"):
                numeric_data *= config["numeric_multiply"]
            processed_data = numeric_data
        except Exception as e:
            errors.append(f"Error processing numeric: {e}")
            log_messages.append(f"Detailed numeric error: {e}")

    # --- Post-processing ---
    if config.get("sort_output"):
        try:
            processed_data.sort()
        except:
            log_messages.append("Warning: Could not sort output data.")

    if errors:
        log_messages.append(f"Found {len(errors)} errors during processing.")

    log_messages.append("Data processing complete.")

    return processed_data, errors, log_messages


# AI Response:
def complicated_data_processor(data, config):
    """
    This function performs a series of data transformations and calculations based on a provided configuration.

    Args:
        data: A list of dictionaries, where each dictionary represents a data record.
        config: A dictionary containing configuration parameters that control the processing steps.

    Returns:
        A list of dictionaries, representing the processed data.  Returns an empty list if input data is invalid.
    """

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        print("Error: Input data must be a list of dictionaries.")
        return []

    if not isinstance(config, dict):
        print("Error: Input config must be a dictionary.")
        return []

    processed_data = []

    for record in data:
        processed_record = record.copy() # Create a copy to avoid modifying the original

        # Step 1: Data Type Conversions
        if config.get("convert_types", False):
            for field, target_type in config.get("type_mapping", {}).items():
                if field in processed_record:
                    try:
                        if target_type == "int":
                            processed_record[field] = int(processed_record[field])
                        elif target_type == "float":
                            processed_record[field] = float(processed_record[field])
                        elif target_type == "str":
                            processed_record[field] = str(processed_record[field])
                        elif target_type == "bool":
                            processed_record[field] = bool(processed_record[field])
                        else:
                            print(f"Warning: Unknown target type '{target_type}' for field '{field}'. Skipping conversion.")
                    except (ValueError, TypeError) as e:
                        print(f"Error converting field '{field}' to type '{target_type}': {e}. Skipping conversion.")

        # Step 2: Data Filtering
        if config.get("filter_data", False):
            for field, condition in config.get("filter_conditions", {}).items():
                if field in processed_record:
                    operator = condition.get("operator")
                    value = condition.get("value")

                    if operator == "equal":
                        if processed_record[field] != value:
                            processed_record = None  # Mark for removal
                            break  # Skip further processing of this record
                    elif operator == "greater_than":
                        try:
                            if processed_record[field] <= value:
                                processed_record = None
                                break
                        except TypeError:
                            print(f"Warning: Cannot compare '{field}' using 'greater_than'. Skipping record.")
                            processed_record = None
                            break
                    elif operator == "less_than":
                        try:
                            if processed_record[field] >= value:
                                processed_record = None
                                break
                        except TypeError:
                            print(f"Warning: Cannot compare '{field}' using 'less_than'. Skipping record.")
                            processed_record = None
                            break
                    elif operator == "contains":
                         if value not in str(processed_record[field]):
                             processed_record = None
                             break

                    else:
                        print(f"Warning: Unknown filter operator '{operator}' for field '{field}'. Skipping filtering.")

        # Step 3: Data Aggregation (example - finding sum of a field)
        if config.get("aggregate_data", False):
            aggregation_field = config.get("aggregation_field")
            if aggregation_field in processed_record:
                try:
                    processed_record["aggregated_value"] = sum([record.get(aggregation_field, 0) for record in data])
                except TypeError:
                    print(f"Warning: Cannot sum field '{aggregation_field}'. Skipping aggregation.")
                    processed_record["aggregated_value"] = None # or some other default value
            else:
                processed_record["aggregated_value"] = None # or some other default value

        # Step 4: Data Transformation (example - calculate a new field)
        if config.get("transform_data", False):
            for new_field, formula in config.get("transformations", {}).items():
                try:
                    # Replace field names in the formula with their values.  Very basic evalutation, use with caution in production.
                    # A more robust solution would use the 'ast' module for safe expression evaluation.
                    safe_formula = formula
                    for field in record:
                        safe_formula = safe_formula.replace(field, str(record.get(field,0))) #replace all field names by their values
                    processed_record[new_field] = eval(safe_formula)

                except (NameError, TypeError, ZeroDivisionError) as e:
                    print(f"Error evaluating formula for '{new_field}': {e}. Setting to None.")
                    processed_record[new_field] = None
                except SyntaxError as e:
                    print(f"Syntax Error in the formula: {e}. Setting to None")
                    processed_record[new_field] = None

        # Step 5: Data Enrichment (example - adding constant value)
        if config.get("enrich_data", False):
            enrichment_field = config.get("enrichment_field")
            enrichment_value = config.get("enrichment_value")
            processed_record[enrichment_field] = enrichment_value


        # Step 6: Data Renaming
        if config.get("rename_fields", False):
            for old_name, new_name in config.get("field_renames", {}).items():
                if old_name in processed_record:
                    processed_record[new_name] = processed_record.pop(old_name)


        if processed_record is not None:
            processed_data.append(processed_record)

    return processed_data


# AI Response:
def process_data_and_generate_report(data, config, logging_enabled=True, report_type="detailed", output_format="csv"):
    """
    Processes a large dataset, performs various calculations and aggregations based on the provided configuration,
    and generates a report in the specified format. Includes extensive error handling and logging capabilities.

    Args:
        data (list of dict): A list of dictionaries representing the data to be processed. Each dictionary
                             should have a consistent structure, as defined in the configuration.
        config (dict): A dictionary containing configuration parameters for data processing and report generation.
                       This includes fields like:
                           - 'fields_to_include': A list of field names to include in the report.
                           - 'aggregation_fields': A list of field names to use for aggregation.
                           - 'filter_criteria': A dictionary specifying filtering conditions (e.g., {'field': 'status', 'value': 'active'}).
                           - 'calculation_rules': A list of dictionaries specifying calculation rules (e.g., {'field': 'price', 'operation': 'multiply', 'factor': 1.1}).
                           - 'report_title': The title of the report.
        logging_enabled (bool, optional): A flag indicating whether logging should be enabled. Defaults to True.
        report_type (str, optional): The type of report to generate ("summary" or "detailed"). Defaults to "detailed".
        output_format (str, optional): The output format of the report ("csv", "json", "text"). Defaults to "csv".

    Returns:
        str: The generated report as a string, or None if an error occurred.
    """

    try:
        import datetime
        import json
        import csv

        def log_message(message):
            if logging_enabled:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] - {message}")

        log_message("Starting data processing and report generation...")

        # Data Validation and Filtering
        if not isinstance(data, list):
            log_message("Error: Input data must be a list.")
            return None

        if not isinstance(config, dict):
            log_message("Error: Configuration must be a dictionary.")
            return None

        filtered_data = data.copy() # Create a copy to avoid modifying the original data

        if 'filter_criteria' in config:
            filter_criteria = config['filter_criteria']
            if isinstance(filter_criteria, dict):
                filtered_data = [item for item in filtered_data if item.get(filter_criteria.get('field')) == filter_criteria.get('value')]
                log_message(f"Data filtered based on criteria: {filter_criteria}")
            else:
                log_message("Warning: Invalid filter criteria format. Skipping filtering.")

        # Data Transformation and Calculation
        if 'calculation_rules' in config:
            calculation_rules = config['calculation_rules']
            if isinstance(calculation_rules, list):
                for rule in calculation_rules:
                    field = rule.get('field')
                    operation = rule.get('operation')
                    factor = rule.get('factor')

                    if field and operation and factor:
                        for item in filtered_data:
                            try:
                                value = item.get(field)
                                if isinstance(value, (int, float)):
                                    if operation == 'multiply':
                                        item[field] = value * factor
                                    elif operation == 'add':
                                        item[field] = value + factor
                                    elif operation == 'subtract':
                                        item[field] = value - factor
                                    elif operation == 'divide':
                                        if factor != 0:
                                            item[field] = value / factor
                                        else:
                                            log_message(f"Error: Division by zero attempted for field {field}.")
                                            return None
                                    else:
                                        log_message(f"Warning: Unknown operation '{operation}' for field {field}. Skipping calculation.")
                                else:
                                    log_message(f"Warning: Invalid data type for field {field}. Skipping calculation.")
                            except Exception as e:
                                log_message(f"Error during calculation for field {field}: {e}")
                                return None
                log_message("Data calculations applied based on configured rules.")
            else:
                log_message("Warning: Invalid calculation rules format. Skipping calculations.")

        # Data Aggregation (if report type is summary)
        aggregated_data = {}
        if report_type == "summary":
            if 'aggregation_fields' in config:
                aggregation_fields = config['aggregation_fields']
                if isinstance(aggregation_fields, list):
                    for field in aggregation_fields:
                        aggregated_data[field] = sum([item.get(field, 0) for item in filtered_data])
                    log_message(f"Data aggregated based on fields: {aggregation_fields}")
                else:
                    log_message("Warning: Invalid aggregation fields format. Skipping aggregation.")
            else:
                log_message("Warning: No aggregation fields specified for summary report. Skipping aggregation.")

        # Report Generation
        report_title = config.get('report_title', 'Data Report')
        fields_to_include = config.get('fields_to_include', list(filtered_data[0].keys() if filtered_data else [])) # infer fields if none provided
        if not isinstance(fields_to_include, list):
            log_message("Warning: Invalid fields_to_include format. Using all available fields.")
            fields_to_include = list(filtered_data[0].keys() if filtered_data else [])

        report_string = ""

        if output_format == "csv":
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fields_to_include)
            writer.writeheader()

            if report_type == "detailed":
                writer.writerows(filtered_data)
            elif report_type == "summary":
                writer.writerow(aggregated_data) # Assuming aggregated_data is a single-row dictionary

            report_string = output.getvalue()

        elif output_format == "json":
            if report_type == "detailed":
                report_string = json.dumps(filtered_data, indent=4)
            elif report_type == "summary":
                report_string = json.dumps(aggregated_data, indent=4)

        elif output_format == "text":
            report_string += f"{report_title}\n"
            report_string += "---------------------\n"
            if report_type == "detailed":
                for item in filtered_data:
                    for field in fields_to_include:
                        report_string += f"{field}: {item.get(field, 'N/A')}\n"
                    report_string += "---------------------\n"
            elif report_type == "summary":
                 for field in fields_to_include:
                     report_string += f"{field}: {aggregated_data.get(field, 'N/A')}\n"

        else:
            log_message(f"Error: Unsupported output format: {output_format}")
            return None

        log_message("Report generation complete.")
        return report_string

    except Exception as e:
        log_message(f"An unexpected error occurred: {e}")
        return None


# AI Response:
def process_data_and_generate_report(data, config, transformations=None, anomaly_threshold=3.0, missing_value_strategy="mean", output_format="text"):
    """
    Processes a dataset, performs various analyses, and generates a report in a specified format.

    Args:
        data: A list of dictionaries, where each dictionary represents a data point.  Each dictionary should have the same keys.
        config: A dictionary containing configuration parameters for the analysis and reporting.
                Required keys might include: 'report_title', 'target_variable', 'features_to_include', 'date_column'.
        transformations: An optional list of functions to apply to the data before analysis. Each function should take a single data point (dictionary) as input and return the modified data point. Defaults to None.
        anomaly_threshold: A numerical threshold for identifying outliers in the target variable. Data points with a Z-score exceeding this threshold are considered anomalies. Defaults to 3.0.
        missing_value_strategy: A string indicating how to handle missing values. Options are "mean", "median", or "drop". Defaults to "mean".
        output_format: A string specifying the desired output format for the report. Options are "text", "csv", or "json". Defaults to "text".

    Returns:
        A string containing the generated report, formatted according to the specified output format.

    Raises:
        ValueError: If the input data is invalid or the configuration is incomplete.
        TypeError: If the data types of the input arguments are incorrect.
    """

    # Input validation
    if not isinstance(data, list):
        raise TypeError("Data must be a list of dictionaries.")
    if not data:
        raise ValueError("Data cannot be empty.")
    if not isinstance(data[0], dict):
        raise TypeError("Each element of data must be a dictionary.")
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary.")
    if 'report_title' not in config or 'target_variable' not in config or 'features_to_include' not in config:
        raise ValueError("Config must contain 'report_title', 'target_variable', and 'features_to_include'.")
    if transformations is not None and not isinstance(transformations, list):
        raise TypeError("Transformations must be a list of functions.")
    if not isinstance(anomaly_threshold, (int, float)):
        raise TypeError("Anomaly threshold must be a number.")
    if missing_value_strategy not in ["mean", "median", "drop"]:
        raise ValueError("Invalid missing value strategy. Must be 'mean', 'median', or 'drop'.")
    if output_format not in ["text", "csv", "json"]:
        raise ValueError("Invalid output format. Must be 'text', 'csv', or 'json'.")

    report_title = config['report_title']
    target_variable = config['target_variable']
    features_to_include = config['features_to_include']

    # Data Preprocessing
    processed_data = data[:]  # Create a copy to avoid modifying the original data

    # Apply transformations, if any
    if transformations:
        for transform in transformations:
            processed_data = [transform(point) for point in processed_data]

    # Handle missing values
    for feature in features_to_include + [target_variable]:
        values = [point.get(feature) for point in processed_data]
        missing_count = values.count(None) #Count missing value which are represented as None
        if missing_count > 0:
            if missing_value_strategy == "mean":
                valid_values = [v for v in values if v is not None and isinstance(v, (int, float))]
                if valid_values:
                    mean_value = sum(valid_values) / len(valid_values)
                    for point in processed_data:
                        if point.get(feature) is None:
                            point[feature] = mean_value
                else:
                    # If all values are missing or not numeric, replace with 0 to avoid errors
                    for point in processed_data:
                        if point.get(feature) is None:
                            point[feature] = 0

            elif missing_value_strategy == "median":
                valid_values = [v for v in values if v is not None and isinstance(v, (int, float))]
                if valid_values:
                    sorted_values = sorted(valid_values)
                    mid = len(sorted_values) // 2
                    median_value = (sorted_values[mid - 1] + sorted_values[mid]) / 2 if len(sorted_values) % 2 == 0 else sorted_values[mid]
                    for point in processed_data:
                        if point.get(feature) is None:
                            point[feature] = median_value
                else:
                    # If all values are missing or not numeric, replace with 0 to avoid errors
                    for point in processed_data:
                        if point.get(feature) is None:
                            point[feature] = 0

            elif missing_value_strategy == "drop":
                processed_data = [point for point in processed_data if point.get(feature) is not None]

    # Anomaly Detection (based on Z-score)
    target_values = [point[target_variable] for point in processed_data] # Assumes missing values already handled
    mean_target = sum(target_values) / len(target_values)
    std_dev_target = (sum([(x - mean_target) ** 2 for x in target_values]) / len(target_values)) ** 0.5

    anomalies = []
    for i, value in enumerate(target_values):
        z_score = (value - mean_target) / std_dev_target if std_dev_target != 0 else 0  # Handle zero std dev
        if abs(z_score) > anomaly_threshold:
            anomalies.append((i, value, z_score))  # Store index, value, and Z-score

    # Calculate Summary Statistics
    summary_stats = {}
    for feature in features_to_include + [target_variable]:
        values = [point[feature] for point in processed_data if isinstance(point[feature], (int, float))] # Filter out non-numeric values
        if values:
            summary_stats[feature] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        else:
            summary_stats[feature] = {'mean': None, 'min': None, 'max': None, 'count': 0}

    # Generate Report
    report = f"Report Title: {report_title}\n\n"
    report += "Summary Statistics:\n"
    for feature, stats in summary_stats.items():
        report += f"  Feature: {feature}\n"
        report += f"    Mean: {stats['mean']}\n"
        report += f"    Min: {stats['min']}\n"
        report += f"    Max: {stats['max']}\n"
        report += f"    Count: {stats['count']}\n\n"

    report += "Anomalies (Z-score > {}):\n".format(anomaly_threshold)
    if anomalies:
        for index, value, z_score in anomalies:
            report += f"  Index: {index}, Value: {value}, Z-score: {z_score}\n"
    else:
        report += "  No anomalies found.\n"

    if output_format == "csv":
        import csv
        import io
        output = io.StringIO()
        fieldnames = features_to_include + [target_variable]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_data)
        report = output.getvalue()
    elif output_format == "json":
        import json
        report = json.dumps({"report_title": report_title, "summary_statistics": summary_stats, "anomalies": anomalies, "data": processed_data})

    return report


# AI Response:
def process_data_and_generate_report(data, config):
    """
    Processes a large dataset, performs various calculations and transformations based on configuration,
    and generates a comprehensive report with different sections and visualizations (simulated).

    Args:
        data: A list of dictionaries representing the dataset. Each dictionary contains various fields.
        config: A dictionary containing configuration parameters for processing and report generation.

    Returns:
        A string representing the generated report.
    """

    report = ""

    # --- Section 1: Data Overview ---
    report += "--- Data Overview ---\n"
    report += f"Total number of records: {len(data)}\n"

    # Basic statistics (simulated)
    numeric_fields = config.get("numeric_fields", [])
    if numeric_fields:
        report += "\nNumeric Field Statistics:\n"
        for field in numeric_fields:
            values = [item.get(field) for item in data if isinstance(item.get(field), (int, float))]
            if values:
                report += f"  {field}:\n"
                report += f"    Minimum: {min(values)}\n"
                report += f"    Maximum: {max(values)}\n"
                report += f"    Average: {sum(values) / len(values)}\n" # Simple Average

    # --- Section 2: Data Cleaning and Transformation ---
    report += "\n--- Data Cleaning and Transformation ---\n"

    # Simulate data cleaning (e.g., handling missing values, data type conversions)
    missing_value_strategy = config.get("missing_value_strategy", "remove")  # "remove" or "impute"
    fields_to_clean = config.get("fields_to_clean", [])

    cleaned_data = []
    for item in data:
        new_item = item.copy()
        for field in fields_to_clean:
            if field not in new_item or new_item[field] is None:
                if missing_value_strategy == "remove":
                    new_item = None # effectively removes the record
                    break
                elif missing_value_strategy == "impute":
                    new_item[field] = config.get("impute_value", 0)  # Default imputation value

        if new_item is not None:
            cleaned_data.append(new_item)
    report += f"Number of records after cleaning: {len(cleaned_data)}\n"

    # Simulate data transformation (e.g., feature engineering)
    transformation_rules = config.get("transformation_rules", {})
    transformed_data = []
    for item in cleaned_data:
        new_item = item.copy()
        for new_field, rule in transformation_rules.items():
            if isinstance(rule, str) and rule.startswith("lambda"):
                try:
                    # Simulate evaluating a lambda function (very unsafe in real scenarios!)
                    # Consider using safer alternatives like defining named functions or a dictionary mapping rules.
                    new_item[new_field] = eval(rule)(item)  #DO NOT DO THIS IN REAL CODE
                except Exception as e:
                    report += f"Error applying transformation rule for {new_field}: {e}\n"

        transformed_data.append(new_item)

    # --- Section 3: Analysis and Insights ---
    report += "\n--- Analysis and Insights ---\n"

    # Simulated analysis: calculate aggregated metrics
    group_by_field = config.get("group_by_field", None)
    if group_by_field:
        report += f"Aggregated Metrics by {group_by_field}:\n"
        grouped_data = {}
        for item in transformed_data:
            group_value = item.get(group_by_field)
            if group_value not in grouped_data:
                grouped_data[group_value] = []
            grouped_data[group_value].append(item)

        for group, group_items in grouped_data.items():
            report += f"  Group: {group}\n"
            # Calculate simple statistics per group
            if numeric_fields:
                for field in numeric_fields:
                    values = [item.get(field) for item in group_items if isinstance(item.get(field), (int, float))]
                    if values:
                        report += f"    {field}:\n"
                        report += f"      Average: {sum(values) / len(values)}\n"

    # --- Section 4: Visualizations (Simulated) ---
    report += "\n--- Visualizations (Simulated) ---\n"
    report += "Placeholder for charts and graphs generated from the data.\n"
    if config.get("include_scatter_plot", False) :
        report += "  Simulated scatter plot: X-axis: Field A, Y-axis: Field B\n"
    if config.get("include_histogram", False) :
        report += "  Simulated histogram: Distribution of Field C\n"

    # --- Section 5: Conclusion ---
    report += "\n--- Conclusion ---\n"
    report += "This report provides an overview of the data, key insights, and potential areas for further investigation.\n"

    return report


# AI Response:
def process_data(data, config):
    """
    This function processes a large dataset based on the provided configuration.
    It performs various data cleaning, transformation, and analysis steps.

    Args:
        data: A list of dictionaries representing the dataset. Each dictionary
              represents a single data point.
        config: A dictionary containing configuration parameters for the
                processing steps.

    Returns:
        A dictionary containing the processed data and analysis results.
    """

    processed_data = []
    analysis_results = {}

    # Data Cleaning
    if config.get("clean_data", True):
        print("Cleaning data...")
        for item in data:
            cleaned_item = {}
            for key, value in item.items():
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    if config.get("lowercase", False):
                        cleaned_value = cleaned_value.lower()
                    cleaned_item[key] = cleaned_value
                elif isinstance(value, (int, float)):
                    cleaned_item[key] = value
                else:
                    cleaned_item[key] = value  # Keep other data types as is
            processed_data.append(cleaned_item)
    else:
        processed_data = data.copy() # Create a copy to avoid modifying original
        print("Skipping data cleaning.")


    # Data Transformation
    if config.get("transform_data", False):
        print("Transforming data...")
        transformation_type = config.get("transformation_type", "default")
        if transformation_type == "default":
            for item in processed_data:
                try:
                    item["transformed_value"] = item["numeric_value"] * 2 # Example transformation
                except KeyError:
                    item["transformed_value"] = None
        elif transformation_type == "custom":
            transformation_function = config.get("transformation_function")
            if transformation_function:
                for item in processed_data:
                    try:
                        item["transformed_value"] = transformation_function(item)
                    except Exception as e:
                        print(f"Error applying custom transformation: {e}")
                        item["transformed_value"] = None
        else:
            print("Invalid transformation type.")

    # Data Filtering
    if config.get("filter_data", False):
        print("Filtering data...")
        filter_criteria = config.get("filter_criteria")
        if filter_criteria:
            filtered_data = []
            for item in processed_data:
                try:
                    if filter_criteria(item):
                        filtered_data.append(item)
                except Exception as e:
                    print(f"Error applying filter criteria: {e}")
            processed_data = filtered_data
        else:
            print("No filter criteria provided.")

    # Data Aggregation
    if config.get("aggregate_data", False):
        print("Aggregating data...")
        aggregation_key = config.get("aggregation_key")
        if aggregation_key:
            aggregated_data = {}
            for item in processed_data:
                key = item.get(aggregation_key)
                if key:
                    if key not in aggregated_data:
                        aggregated_data[key] = []
                    aggregated_data[key].append(item)
            analysis_results["aggregated_data"] = aggregated_data
        else:
            print("No aggregation key provided.")

    # Data Analysis
    if config.get("analyze_data", False):
        print("Analyzing data...")
        # Example analysis: Calculate the average of a numeric field
        numeric_field = config.get("numeric_field")
        if numeric_field:
            total = 0
            count = 0
            for item in processed_data:
                try:
                    value = item.get(numeric_field)
                    if isinstance(value, (int, float)):
                        total += value
                        count += 1
                except Exception as e:
                    print(f"Error accessing numeric field: {e}")
            if count > 0:
                average = total / count
                analysis_results["average_" + numeric_field] = average
            else:
                analysis_results["average_" + numeric_field] = None

    # Additional analysis example : Calculate the frequency of a categorical field
    if config.get("analyze_data", False):
        categorical_field = config.get("categorical_field")
        if categorical_field:
            frequencies = {}
            for item in processed_data:
                try:
                    value = item.get(categorical_field)
                    if value:
                        if value not in frequencies:
                            frequencies[value] = 0
                        frequencies[value] += 1
                except Exception as e:
                    print(f"Error accessing categorical field: {e}")
            analysis_results["frequencies_" + categorical_field] = frequencies

    # Feature Engineering
    if config.get("feature_engineering", False):
        print("Performing feature engineering...")
        new_feature_name = config.get("new_feature_name")
        feature_function = config.get("feature_function")
        if new_feature_name and feature_function:
            for item in processed_data:
                try:
                    item[new_feature_name] = feature_function(item)
                except Exception as e:
                    print(f"Error creating new feature: {e}")
                    item[new_feature_name] = None

    return {"processed_data": processed_data, "analysis_results": analysis_results}


# AI Response:
def process_data(data, operation_type="basic", scaling_factor=1.0, custom_function=None, data_validation=True, logging_enabled=False, error_handling="strict", missing_value_handling="impute_mean", imputation_value=0, outlier_removal="zscore", zscore_threshold=3, window_size=5, aggregation_function="mean", rolling_calculation="mean", date_column=None, date_format="%Y-%m-%d", categorical_encoding="onehot", columns_to_encode=None, text_processing="lowercase", stopwords_removal=False, stemmer="porter", vectorization="tfidf", n_features=100, model_type="linear_regression", regularization="l1", alpha=0.1, cross_validation=True, n_folds=5, feature_selection="select_k_best", k_best=10, performance_metric="accuracy", confidence_interval=0.95, parallel_processing=False, num_cores=2, save_results=False, output_path="results.csv", database_connection=None, query=None, table_name=None, email_notifications=False, email_address="user@example.com", api_integration=False, api_endpoint=None, api_key=None):
    """
    A comprehensive function for processing and analyzing data with various options for data cleaning, transformation, feature engineering, modeling, and reporting.

    Args:
        data (list or pandas.DataFrame): The input data to be processed.  Can be a list of lists, list of dictionaries, or a pandas DataFrame.
        operation_type (str): The primary operation to perform on the data ("basic", "advanced", "machine_learning").
        scaling_factor (float): A scaling factor to apply to numerical data.
        custom_function (callable): A custom function to apply to each element or row of the data. Must be pickleable if parallel_processing is True.
        data_validation (bool): Whether to perform data validation checks (e.g., missing values, data types).
        logging_enabled (bool): Whether to enable logging of the processing steps.
        error_handling (str): How to handle errors ("strict", "ignore", "report").
        missing_value_handling (str): Method for handling missing values ("drop", "impute_mean", "impute_median", "impute_constant").
        imputation_value (int or float): Value to use for constant imputation of missing values.
        outlier_removal (str): Method for outlier removal ("zscore", "iqr", None).
        zscore_threshold (int or float): Z-score threshold for outlier removal.
        window_size (int): Window size for rolling calculations.
        aggregation_function (str): Function to use for aggregation (e.g., "mean", "sum", "min", "max").
        rolling_calculation (str): Type of rolling calculation ("mean", "sum", "std").
        date_column (str): Name of the column containing date information. Required for date-based operations.
        date_format (str): Format string for parsing dates.
        categorical_encoding (str): Method for encoding categorical features ("onehot", "label", "ordinal").
        columns_to_encode (list): List of categorical columns to encode. If None, all object type columns will be encoded.
        text_processing (str): Type of text processing to perform ("lowercase", "uppercase", None).
        stopwords_removal (bool): Whether to remove stopwords from text data.
        stemmer (str): Type of stemmer to use ("porter", "lancaster").
        vectorization (str): Method for vectorizing text data ("tfidf", "count").
        n_features (int): Maximum number of features to retain after vectorization.
        model_type (str): Type of machine learning model to use ("linear_regression", "logistic_regression", "decision_tree", "random_forest").
        regularization (str): Type of regularization to apply ("l1", "l2", None).
        alpha (float): Regularization strength.
        cross_validation (bool): Whether to perform cross-validation.
        n_folds (int): Number of folds for cross-validation.
        feature_selection (str): Method for feature selection ("select_k_best", "rfe", None).
        k_best (int): Number of top features to select (for select_k_best).
        performance_metric (str): Metric to evaluate model performance (e.g., "accuracy", "precision", "recall", "f1", "rmse", "r2").
        confidence_interval (float): Confidence level for calculating confidence intervals.
        parallel_processing (bool): Whether to use parallel processing.
        num_cores (int): Number of cores to use for parallel processing.
        save_results (bool): Whether to save the processed data to a file.
        output_path (str): Path to save the processed data.
        database_connection (str): Connection string for connecting to a database.
        query (str): SQL query to execute on the database.
        table_name (str): Name of the table to read from or write to in the database.
        email_notifications (bool): Whether to send email notifications upon completion.
        email_address (str): Email address to send notifications to.
        api_integration (bool): Whether to integrate with an external API.
        api_endpoint (str): URL of the external API endpoint.
        api_key (str): API key for accessing the external API.

    Returns:
        pandas.DataFrame: The processed data as a pandas DataFrame.  Returns None if an error occurs and error_handling is set to "strict".
    """

    import pandas as pd
    import numpy as np
    import logging
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, LancasterStemmer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, RFE
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
    from scipy import stats
    import multiprocessing
    from functools import partial

    try:
        # Initialize logging
        if logging_enabled:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info("Data processing started.")

        # Convert data to pandas DataFrame
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a list or pandas DataFrame.")

        # Data Validation
        if data_validation:
            if data.isnull().sum().sum() > 0:
                logging.warning("Missing values found in the dataset.")
            if not all(isinstance(col, str) for col in data.columns):
                logging.warning("Column names should be strings.")

        # Missing Value Handling
        if missing_value_handling == "drop":
            data = data.dropna()
            if logging_enabled:
                logging.info("Dropped rows with missing values.")
        elif missing_value_handling == "impute_mean":
            data = data.fillna(data.mean(numeric_only=True))
            if logging_enabled:
                logging.info("Imputed missing values with the mean.")
        elif missing_value_handling == "impute_median":
            data = data.fillna(data.median(numeric_only=True))
            if logging_enabled:
                logging.info("Imputed missing values with the median.")
        elif missing_value_handling == "impute_constant":
            data = data.fillna(imputation_value)
            if logging_enabled:
                logging.info(f"Imputed missing values with constant value: {imputation_value}.")

        # Basic Operation
        if operation_type == "basic":
            data = data * scaling_factor
            if logging_enabled:
                logging.info(f"Applied scaling factor: {scaling_factor}.")

        # Advanced Operations
        elif operation_type == "advanced":
            if custom_function:
                if parallel_processing:
                    num_processes = num_cores if num_cores else multiprocessing.cpu_count()
                    with multiprocessing.Pool(processes=num_processes) as pool:
                        data = pd.DataFrame(pool.map(custom_function, data.values.tolist()), columns=data.columns)

                    if logging_enabled:
                        logging.info("Applied custom function with parallel processing.")

                else:
                    data = data.apply(custom_function, axis=1)
                    if logging_enabled:
                        logging.info("Applied custom function.")
            else:
                logging.warning("No custom function provided for advanced operation.")

            if date_column:
                data[date_column] = pd.to_datetime(data[date_column], format=date_format)
                data['year'] = data[date_column].dt.year
                data['month'] = data[date_column].dt.month
                if logging_enabled:
                    logging.info("Extracted year and month from date column.")

            if window_size > 0:
                if rolling_calculation == "mean":
                    data['rolling_mean'] = data.iloc[:, 0].rolling(window=window_size).mean()
                elif rolling_calculation == "sum":
                    data['rolling_sum'] = data.iloc[:, 0].rolling(window=window_size).sum()
                elif rolling_calculation == "std":
                    data['rolling_std'] = data.iloc[:, 0].rolling(window=window_size).std()
                if logging_enabled:
                    logging.info(f"Applied rolling {rolling_calculation} with window size: {window_size}.")

        # Outlier Removal
        if outlier_removal == "zscore":
            for col in data.select_dtypes(include=np.number).columns:
                z = np.abs(stats.zscore(data[col]))
                data = data[(z < zscore_threshold)]
            if logging_enabled:
                logging.info(f"Removed outliers based on Z-score with threshold: {zscore_threshold}.")
        elif outlier_removal == "iqr":
            for col in data.select_dtypes(include=np.number).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
            if logging_enabled:
                logging.info("Removed outliers based on IQR.")

        # Categorical Encoding
        if categorical_encoding:
            if columns_to_encode is None:
                columns_to_encode = data.select_dtypes(include=['object']).columns.tolist()

            if categorical_encoding == "onehot":
                data = pd.get_dummies(data, columns=columns_to_encode)
                if logging_enabled:
                    logging.info(f"Applied one-hot encoding to columns: {columns_to_encode}.")
            elif categorical_encoding == "label":
                for col in columns_to_encode:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                if logging_enabled:
                    logging.info(f"Applied label encoding to columns: {columns_to_encode}.")
            elif categorical_encoding == "ordinal": #Simplified for brevity, real implementation needs mapping
                 for col in columns_to_encode:
                    data[col] = data[col].astype('category')
                    data[col] = data[col].cat.codes
                 if logging_enabled:
                    logging.info(f"Applied ordinal encoding to columns: {columns_to_encode}.")

        # Text Processing
        if text_processing:
            for col in data.select_dtypes(include=['object']).columns:
                if text_processing == "lowercase":
                    data[col] = data[col].str.lower()
                elif text_processing == "uppercase":
                    data[col] = data[col].str.upper()
            if logging_enabled:
                logging.info(f"Applied text processing: {text_processing}.")

        if stopwords_removal:
            stop_words = set(stopwords.words('english'))
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
            if logging_enabled:
                logging.info("Removed stopwords from text columns.")

        if stemmer:
            if stemmer == "porter":
                stemmer_obj = PorterStemmer()
            elif stemmer == "lancaster":
                stemmer_obj = LancasterStemmer()

            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].apply(lambda x: ' '.join([stemmer_obj.stem(word) for word in x.split()]))
            if logging_enabled:
                logging.info(f"Applied {stemmer} stemming to text columns.")

        if vectorization:
            for col in data.select_dtypes(include=['object']).columns:
                if vectorization == "tfidf":
                    vectorizer = TfidfVectorizer(max_features=n_features)
                elif vectorization == "count":
                    vectorizer = CountVectorizer(max_features=n_features)

                vectorized_data = vectorizer.fit_transform(data[col]).toarray()
                vectorized_df = pd.DataFrame(vectorized_data, columns=[f'{col}_{i}' for i in range(vectorized_data.shape[1])])
                data = pd.concat([data.drop(col, axis=1), vectorized_df], axis=1)
            if logging_enabled:
                logging.info(f"Applied {vectorization} vectorization with {n_features} features.")

        # Machine Learning Operations
        if operation_type == "machine_learning":
            # Separate features and target (assuming the last column is the target)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            # Feature Scaling
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Feature Selection
            if feature_selection == "select_k_best":
                selector = SelectKBest(k=k_best)
                X = selector.fit_transform(X, y)
                if logging_enabled:
                    logging.info(f"Applied SelectKBest feature selection with k={k_best}.")
            elif feature_selection == "rfe":
                if model_type == "linear_regression":
                     model = LinearRegression()
                elif model_type == "logistic_regression":
                    model = LogisticRegression(penalty=regularization, C=1/alpha, solver='liblinear') # Added solver
                elif model_type == "decision_tree":
                    model = DecisionTreeClassifier()
                elif model_type == "random_forest":
                    model = RandomForestClassifier()

                selector = RFE(model, n_features_to_select=k_best)
                X = selector.fit_transform(X, y)
                if logging_enabled:
                    logging.info(f"Applied RFE feature selection with k={k_best}.")

            # Model Training
            if model_type == "linear_regression":
                model = LinearRegression()
            elif model_type == "logistic_regression":
                model = LogisticRegression(penalty=regularization, C=1/alpha, solver='liblinear')  # Added solver
            elif model_type == "decision_tree":
                model = DecisionTreeClassifier()
            elif model_type == "random_forest":
                model = RandomForestClassifier()

            # Cross-validation
            if cross_validation:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)  # Added random_state
                cv_scores = cross_val_score(model, X, y, cv=kf, scoring=performance_metric)
                if logging_enabled:
                    logging.info(f"Performed cross-validation with {n_folds} folds.")
                print(f"Cross-validation {performance_metric}: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Added random_state
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate model
            if performance_metric == "accuracy":
                score = accuracy_score(y_test, y_pred)
            elif performance_metric == "precision":
                score = precision_score(y_test, y_pred, average='weighted')
            elif performance_metric == "recall":
                score = recall_score(y_test, y_pred, average='weighted')
            elif performance_metric == "f1":
                score = f1_score(y_test, y_pred, average='weighted')
            elif performance_metric == "rmse":
                score = np.sqrt(mean_squared_error(y_test, y_pred))
            elif performance_metric == "r2":
                score = r2_score(y_test, y_pred)
            else:
                score = 0.0
                logging.warning(f"Invalid performance metric: {performance_metric}")

            print(f"{performance_metric}: {score:.4f}")

            if logging_enabled:
                logging.info(f"Model training completed. {performance_metric}: {score:.4f}")
                logging.info("Machine Learning operations completed.")

        # Save Results
        if save_results:
            data.to_csv(output_path, index=False)
            if logging_enabled:
                logging.info(f"Saved processed data to: {output_path}.")

        # Database Integration
        if database_connection and query:
            # This would require a real database connection setup and SQL execution
            # This is a placeholder - replace with actual database interaction code.
            print("Simulating database query execution...")
            if logging_enabled:
                 logging.info("Simulated database query execution.")
        elif database_connection and table_name:
            # placeholder - replace with writing data to database logic
            print(f"Simulating writing to table {table_name} in the database.")
            if logging_enabled:
                logging.info(f"Simulated writing data to database table {table_name}.")

        # API Integration
        if api_integration and api_endpoint and api_key:
            # This would require a real API call with authentication
            # This is a placeholder - replace with actual API call code.
            print(f"Simulating API call to {api_endpoint} with key {api_key[:4]}...") #Show first 4 characters only.
            if logging_enabled:
                logging.info(f"Simulated API call to {api_endpoint}.")

        # Email Notifications
        if email_notifications and email_address:
            # This would require a real email sending setup
            # This is a placeholder - replace with actual email sending code.
            print(f"Sending notification email to: {email_address}")
            if logging_enabled:
                logging.info(f"Sending notification email to: {email_address}.")

        if logging_enabled:
            logging.info("Data processing completed successfully.")

        return data

    except Exception as e:
        if error_handling == "strict":
            print(f"An error occurred: {e}")
            if logging_enabled:
                logging.error(f"An error occurred: {e}")
            return None
        elif error_handling == "ignore":
            print(f"An error occurred, but processing continues: {e}")
            if logging_enabled:
                logging.warning(f"An error occurred, but processing continues: {e}")
            return data  # Return the data as is (possibly partially processed)
        elif error_handling == "report":
            print(f"An error occurred: {e}")
            if logging_enabled:
                logging.error(f"An error occurred: {e}")
            #  Logic to send an error report to a designated location would go here.
            print("Error reported to administrator.")
            return None # Or return the data as is, depending on desired behavior.
        else:
            print(f"An unexpected error occurred and invalid error_handling type was specified: {e}")
            return None

