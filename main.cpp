#include "app.h"

#include <iostream>

int main() {

	VkApp* app = new VkApp(800, 600, "Testing", true);

	try {
		app->run();
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
	}

	delete app;

	return 0;
}