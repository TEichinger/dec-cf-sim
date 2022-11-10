from unittest import TestLoader, TestResult



def test_folder(folder_name):
	""" Bundle all tests in the folder_name into a test suite an run it. """
	test_loader = TestLoader()
	test_suite = test_loader.discover(folder_name)
	number_of_tests = test_suite.countTestCases()
	print("Test {} consisting of a total of {} tests.".format(folder_name, number_of_tests))
	test_result = TestResult()
	result = test_suite.run(test_result)
	return result
	





def main():
	# search for all unit tests in the subdirectories
	# and create a test suite out of them

	# test mobility_models
	result = test_folder("test_mobility_models")
	print(result)

	# test algorithms
	result = test_folder("test_algorithms")
	print(result)

	
	
	
if __name__ == "__main__":
	main()