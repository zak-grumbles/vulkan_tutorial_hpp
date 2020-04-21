#define NOMINMAX
#include "app.h"

#include <iostream>
#include <set>

const int kMaxFramesInFlight = 2;

const std::vector<const char*> kValidationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> kDeviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

VkResult create_dbg_utils_messenger(
	VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT* create_info,
	const VkAllocationCallbacks* allocator,
	VkDebugUtilsMessengerEXT* debug_messenger
) {

	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,
		"vkCreateDebugUtilsMessengerEXT");

	if (func != nullptr) {
		return func(instance, create_info, allocator, debug_messenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void destroy_dbg_utils_messenger(
	VkInstance instance,
	VkDebugUtilsMessengerEXT dbg_messenger,
	const VkAllocationCallbacks* allocator
) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

	if (func != nullptr) {
		func(instance, dbg_messenger, allocator);
	}
}

VkApp::VkApp(int width, int height, std::string title, bool validation_enabled /*= false*/) {
	validation_enabled_ = validation_enabled;

	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window_ = glfwCreateWindow(width, height, title.c_str(),
		nullptr, nullptr);

	init_vulkan();
}

VkApp::~VkApp() {

	for (vk::ImageView& view : swapchain_image_views_) {
		device_.destroyImageView(view);
	}

	device_.destroySwapchainKHR(swapchain_);

	device_.destroy();

	destroy_dbg_utils_messenger(instance_, dbg_messenger_, nullptr);

	instance_.destroySurfaceKHR(surface_);

	instance_.destroy();

	glfwDestroyWindow(window_);
	window_ = nullptr;

	glfwTerminate();
}

void VkApp::run() {
	while (!glfwWindowShouldClose(window_)) {
		glfwPollEvents();
	}
}

void VkApp::init_vulkan() {
	init_instance();

	if (validation_enabled_) {
		init_dbg_messenger();
	}

	init_surface();
	select_physical_device();
	init_logical_device();
	init_swapchain();
	init_image_views();
}

void VkApp::init_instance() {
	vk::ApplicationInfo app_info(
		"LunarG Tutorial",
		VK_MAKE_VERSION(1, 0, 0),
		"No Engine",
		VK_MAKE_VERSION(1, 0, 0),
		VK_API_VERSION_1_0
	);

	auto required_exts = get_required_exts();

	uint32_t layer_count = 0;
	const char* const* layers = nullptr;

	if (validation_enabled_) {
		layer_count = static_cast<uint32_t>(kValidationLayers.size());
		layers = kValidationLayers.data();
	}

	vk::InstanceCreateInfo info(
		{},
		&app_info,
		layer_count,
		layers,
		static_cast<uint32_t>(required_exts.size()),
		required_exts.data()
	);

	instance_ = vk::createInstance(info);
}

std::vector<const char*> VkApp::get_required_exts() const noexcept {
	uint32_t glfw_ext_count = 0;
	const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

	std::vector<const char*> exts(glfw_exts, glfw_exts + glfw_ext_count);

	if (validation_enabled_) {
		exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return exts;
}

void VkApp::init_dbg_messenger() {

	VkDebugUtilsMessengerCreateInfoEXT info = {};
	info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	info.pfnUserCallback = dbg_callback;

	VkDebugUtilsMessengerEXT c_messenger;
	if (create_dbg_utils_messenger(instance_, &info, nullptr, &c_messenger) != VK_SUCCESS) {
		throw std::runtime_error("Unable to create debug messenger");
	}

	dbg_messenger_ = vk::DebugUtilsMessengerEXT(c_messenger);
}

VKAPI_ATTR VkBool32 VKAPI_CALL VkApp::dbg_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT severity,
	VkDebugUtilsMessageTypeFlagsEXT type,
	const VkDebugUtilsMessengerCallbackDataEXT* cb_data,
	void* user_data
) {
	std::cerr << "validation layer: " << cb_data->pMessage << std::endl;

	return VK_FALSE;
}

void VkApp::init_surface() {
	VkSurfaceKHR c_surface;

	if (glfwCreateWindowSurface(instance_, window_, nullptr, &c_surface) != VK_SUCCESS) {
		throw std::runtime_error("Unable to create surface");
	}

	surface_ = vk::SurfaceKHR(c_surface);
}

void VkApp::select_physical_device() {
	std::vector<vk::PhysicalDevice> devices = instance_.enumeratePhysicalDevices();

	bool found = false;
	for (const vk::PhysicalDevice& device : devices) {
		if (is_device_suitable(device)) {
			physical_device_ = device;
			queue_family_indices_ = get_queue_families(device);
			found = true;
			break;
		}
	}

	if (!found) {
		throw std::runtime_error("Could not find suitable GPU");
	}
}

bool VkApp::is_device_suitable(vk::PhysicalDevice device) const noexcept {
	QueueFamilies indices = get_queue_families(device);

	bool valid = indices.is_complete();
	valid &= supports_extensions(device);

	if (valid) {
		SwapchainSupport support = get_swapchain_support(device);

		valid &= !support.formats.empty() && !support.present_modes.empty();
	}
	
	return valid;
}

QueueFamilies VkApp::get_queue_families(vk::PhysicalDevice device) const noexcept {
	QueueFamilies families;

	std::vector<vk::QueueFamilyProperties> available_fams = device.getQueueFamilyProperties();

	int i = 0;
	for (const vk::QueueFamilyProperties& family : available_fams) {
		if (family.queueFlags & vk::QueueFlagBits::eGraphics) {
			families.graphics_family = i;
		}

		if (device.getSurfaceSupportKHR(i, surface_)) {
			families.present_family = i;
		}

		if (families.is_complete()) {
			break;
		}

		i++;
	}

	return families;
}

bool VkApp::supports_extensions(vk::PhysicalDevice device) const noexcept {

	std::vector<vk::ExtensionProperties> available_exts = device.enumerateDeviceExtensionProperties();
	
	std::set<std::string> required_exts(kDeviceExtensions.begin(), kDeviceExtensions.end());

	for (const vk::ExtensionProperties& ext : available_exts) {
		required_exts.erase(ext.extensionName);
	}

	return required_exts.empty();
}

SwapchainSupport VkApp::get_swapchain_support(vk::PhysicalDevice device) const noexcept {
	SwapchainSupport support_details;

	support_details.capabilities = device.getSurfaceCapabilitiesKHR(surface_);
	support_details.formats = device.getSurfaceFormatsKHR(surface_);
	support_details.present_modes = device.getSurfacePresentModesKHR(surface_);

	return support_details;
}

void VkApp::init_logical_device() {

	std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
	std::set<uint32_t> unique_queue_families = {
		queue_family_indices_.graphics_family.value(),
		queue_family_indices_.present_family.value()
	};

	float queue_priority = 1.0f;
	for (uint32_t family : unique_queue_families) {
		vk::DeviceQueueCreateInfo queue_info(
			{},
			family,
			1,
			&queue_priority
		);

		queue_create_infos.push_back(queue_info);
	}

	// leave as default intentionally
	vk::PhysicalDeviceFeatures device_features;

	uint32_t layer_count = 0;
	const char* const* layers = nullptr;

	if (validation_enabled_) {
		layer_count = static_cast<uint32_t>(kValidationLayers.size());
		layers = kValidationLayers.data();
	}

	vk::DeviceCreateInfo device_info(
		{},
		static_cast<uint32_t>(queue_create_infos.size()),
		queue_create_infos.data(),
		layer_count,
		layers,
		static_cast<uint32_t>(kDeviceExtensions.size()),
		kDeviceExtensions.data(),
		&device_features
	);

	device_ = physical_device_.createDevice(device_info);

	graphics_queue_ = device_.getQueue(queue_family_indices_.graphics_family.value(), 0);
	present_queue_ = device_.getQueue(queue_family_indices_.present_family.value(), 0);
}

void VkApp::init_swapchain() {
	SwapchainSupport support_details = get_swapchain_support(physical_device_);

	swap_format_ = select_swap_surface_format(support_details.formats);
	swap_present_mode_ = select_swap_present_mode(support_details.present_modes);
	swap_extent_ = select_swap_extent(support_details.capabilities);

	uint32_t img_count = support_details.capabilities.minImageCount + 1;
	if (support_details.capabilities.maxImageCount > 0 && img_count > support_details.capabilities.maxImageCount) {
		img_count = support_details.capabilities.maxImageCount;
	}

	uint32_t queue_family_indices[] = {
		queue_family_indices_.graphics_family.value(),
		queue_family_indices_.present_family.value()
	};
	vk::SharingMode sharing_mode = vk::SharingMode::eExclusive;
	uint32_t queue_index_count = 0;

	if (queue_family_indices_.graphics_family.value() != queue_family_indices_.present_family.value()) {
		sharing_mode = vk::SharingMode::eConcurrent;
		queue_index_count = 2;
	}

	vk::SwapchainCreateInfoKHR sc_info(
		{},
		surface_,
		img_count,
		swap_format_.format,
		swap_format_.colorSpace,
		swap_extent_,
		1,
		vk::ImageUsageFlagBits::eColorAttachment,
		sharing_mode,
		queue_index_count,
		queue_family_indices,
		support_details.capabilities.currentTransform,
		vk::CompositeAlphaFlagBitsKHR::eOpaque,
		swap_present_mode_,
		VK_TRUE
	);

	swapchain_ = device_.createSwapchainKHR(sc_info);
	swapchain_images_ = device_.getSwapchainImagesKHR(swapchain_);
}

vk::SurfaceFormatKHR VkApp::select_swap_surface_format(std::vector<vk::SurfaceFormatKHR> formats) const noexcept {
	vk::SurfaceFormatKHR chosen_format = formats[0];

	for (const vk::SurfaceFormatKHR& format : formats) {
		if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			chosen_format = format;
			break;
		}
	}

	return chosen_format;
}

vk::PresentModeKHR VkApp::select_swap_present_mode(std::vector<vk::PresentModeKHR> modes) const noexcept {
	vk::PresentModeKHR chosen_mode = vk::PresentModeKHR::eFifo;

	for (const vk::PresentModeKHR& mode : modes) {
		if (mode == vk::PresentModeKHR::eMailbox) {
			chosen_mode = mode;
			break;
		}
	}

	return chosen_mode;
}

vk::Extent2D VkApp::select_swap_extent(vk::SurfaceCapabilitiesKHR capabilities) const noexcept {
	vk::Extent2D chosen_extent = capabilities.currentExtent;

	// if set to max, we can chose our own extent
	if (capabilities.currentExtent.width == UINT32_MAX) {

		int width, height;
		glfwGetWindowSize(window_, &width, &height);

		chosen_extent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		chosen_extent.width = std::max(capabilities.minImageExtent.width,
			std::min(capabilities.maxImageExtent.width, chosen_extent.width));

		chosen_extent.height = std::max(capabilities.minImageExtent.height,
			std::min(capabilities.maxImageExtent.height, chosen_extent.height));
	}

	return chosen_extent;
}

void VkApp::init_image_views() {
	swapchain_image_views_.reserve(swapchain_images_.size());

	for (const vk::Image& img : swapchain_images_) {
		vk::ComponentMapping mapping = {};
		mapping.r = vk::ComponentSwizzle::eIdentity;
		mapping.g = vk::ComponentSwizzle::eIdentity;
		mapping.b = vk::ComponentSwizzle::eIdentity;
		mapping.a = vk::ComponentSwizzle::eIdentity;

		vk::ImageSubresourceRange range = {};
		range.aspectMask = vk::ImageAspectFlagBits::eColor;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		vk::ImageViewCreateInfo info(
			{},
			img,
			vk::ImageViewType::e2D,
			swap_format_.format,
			mapping,
			range
		);

		swapchain_image_views_.push_back(device_.createImageView(info));
	}
}