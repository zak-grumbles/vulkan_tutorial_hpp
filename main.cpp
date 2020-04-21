#include "app.h"

int main() {

	VkApp* app = new VkApp(800, 600, "Testing", true);

	app->run();

	delete app;

	return 0;
}