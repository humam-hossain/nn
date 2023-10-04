#if 0

#include <stdio.h>
#include "raylib.h"
#define FACTOR 100

int main()
{
	const int screenWidth = 16*FACTOR;
	const int screenHeight = 9*FACTOR;

	InitWindow(screenWidth, screenHeight, "Hello");
	SetTargetFPS(60);

	while (!WindowShouldClose()) {
		BeginDrawing();
		{
			ClearBackground(RAYWHITE);
			DrawCircle(screenWidth / 2, screenHeight / 2, 100, RED);
		}
		EndDrawing();
	}

	CloseWindow();

	return 0;
}

#endif