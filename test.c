#include "raylib.h"

int main()
{
    const int screenWidth = 800;
    const int screenHeight = 600;

    InitWindow(screenWidth, screenHeight, "hello");
    SetTargetFPS(60);

    while(!WindowShouldClose()){
        BeginDrawing();
        {
            ClearBackground(RAYWHITE);
            DrawText("Hello", screenWidth/2, screenHeight/2, 20, BLUE);
        }
        EndDrawing();
    }

    CloseWindow();

    return 0;
}