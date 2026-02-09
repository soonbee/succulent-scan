<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Inertia\Inertia;
use Inertia\Response;

class DescriptionController extends Controller
{
    public function show(Request $request): Response
    {
        return Inertia::render('Description', [
            'genus' => $request->query('genus', ''),
        ]);
    }
}
